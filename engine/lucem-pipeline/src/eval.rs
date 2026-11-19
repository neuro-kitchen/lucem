//! Runs a graph of ops: [`evaluate`] computes one node's value from its inputs, recursively, each
//! node once (memoized), with the source rasters given by the caller (an image region, a mask
//! drawn by hand, …).
//!
//! For interactive processed views (desktop and `lucem serve --web`), [`evaluate_cached`] adds:
//! - **Incremental recompute** ([`EvalCache`]): uploaded inputs and intermediate node outputs stay
//!   on the device keyed by their structural subgraph and region; editing step `k` or toggling an
//!   op reuses step `k − 1` directly on the GPU without re-reading or re-uploading.
//! - **Pointwise fusion**: consecutive uncached unary pointwise ops (`Rescale`, `Gamma`, …) collapse
//!   into a single [`PointwiseRun`](lucem_ops::core::pointwise::PointwiseRun) kernel launch.
//! - **Per-step timings** ([`EvalTimings`]): host timings always; exact per-step GPU timings in
//!   measuring mode (`LUCEM_PREVIEW_TIMING=1`, native only).
//! - **On-device halo crop** ([`crop_and_download`]): strips halo margins on the GPU before host
//!   read-back so halo pixels never cross the bus.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Mutex;

use lucem_core::cubecl::prelude::Client;
use lucem_ops::core::pointwise::{PointwiseOp, PointwiseRun};
use web_time::Instant;

use crate::ops::{Op, PortType, Value};

/// Pixels a region's halo-padded read bounds are rounded outward to (on all four edges): editing a
/// filter's sigma/radius slightly changes its halo, and rounding to `READ_GRID` lets the new plan
/// reuse the already-uploaded raw input and upstream checkpoints on the device.
pub const READ_GRID: usize = 64;

/// Default device-memory budget for [`EvalCache`] (256 MiB).
pub const DEFAULT_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Rounds `[x0..x1, y0..y1]` outward to multiples of [`READ_GRID`], clamped to `[0..width, 0..height]`.
/// Returns `(rx0, ry0, rw, rh)`.
pub fn align_region(x0: usize, y0: usize, x1: usize, y1: usize, width: usize, height: usize) -> (usize, usize, usize, usize) {
    let rx0 = (x0 / READ_GRID) * READ_GRID;
    let ry0 = (y0 / READ_GRID) * READ_GRID;
    let rx1 = x1.div_ceil(READ_GRID).saturating_mul(READ_GRID).min(width);
    let ry1 = y1.div_ceil(READ_GRID).saturating_mul(READ_GRID).min(height);
    (rx0, ry0, rx1.saturating_sub(rx0), ry1.saturating_sub(ry0))
}

/// One node of a plan.
#[derive(Debug, Clone)]
pub struct PlanNode {
    /// `None`: a source; its value comes from the caller.
    pub op: Option<Op>,
    /// Off: the node passes its first input through (when the types allow it).
    pub enabled: bool,
    /// The node feeding each input port.
    pub inputs: Vec<Option<String>>,
}

/// The graph to run, by node id.
pub type Plan = BTreeMap<String, PlanNode>;

/// Where a processed view evaluation spent its time, in milliseconds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalTimings {
    /// Reading and uploading source rasters (0 when served from [`EvalCache`]).
    pub source_ms: f64,
    /// Launching (and in measuring mode, running) the device ops.
    pub compute_ms: f64,
    /// Cropping on the device and reading back to the host.
    pub readback_ms: f64,
    /// Per-step timings `(label, ms)` when `LUCEM_PREVIEW_TIMING=1` (native only).
    pub steps: Vec<(String, f64)>,
    /// Number of cache hits (source + intermediate nodes reused on the device).
    pub cache_hits: usize,
    /// Number of fused pointwise runs executed.
    pub fused_runs: usize,
    /// Whether at least one source node was reused from the device cache.
    pub input_reused: bool,
    /// Estimated device bytes currently held by the [`EvalCache`].
    pub kept_bytes: usize,
    /// Active byte budget of the [`EvalCache`].
    pub budget_bytes: usize,
    /// Real device bytes currently reserved in CubeCL's pools (0 on wasm).
    pub reserved_bytes: u64,
    /// Cumulative evictions performed by the [`EvalCache`] to stay within budget.
    pub evictions: usize,
}

fn measuring() -> bool {
    !cfg!(target_family = "wasm") && std::env::var_os("LUCEM_PREVIEW_TIMING").is_some()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn value_bytes(v: &Value) -> usize {
    match v {
        Value::Image(a) => a.len() * size_of::<f32>(),
        Value::Mask(a) | Value::Labels(a) => a.len() * size_of::<u32>(),
        Value::Table(rows) => rows.len() * 128,
    }
}

struct CacheInner {
    entries: HashMap<String, (Value, usize)>,
    order: VecDeque<String>,
    total_bytes: usize,
    /// `None`: sized lazily from `lucem_core::memory::device_budget_bytes(client)` on first use.
    max_bytes: Option<usize>,
    evictions: usize,
}

/// Thread-safe LRU cache of device-resident [`Value`]s keyed by structural subgraph + region.
///
/// Shared across processed-view evaluations (in the desktop app's `Store` and on `lucem serve`)
/// so editing step `k` or toggling an op reuses step `k − 1` directly on the GPU.
///
/// By default ([`EvalCache::default`]), its byte budget is sized lazily from the GPU's reported
/// memory (`client.properties().memory.max_memory() / 2`, or `LUCEM_DEVICE_MEMORY`). When entries
/// are evicted to make room before storing a new entry, `client.memory_cleanup()` returns freed
/// CubeCL pages to the driver.
pub struct EvalCache {
    inner: Mutex<CacheInner>,
}

impl Default for EvalCache {
    fn default() -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                order: VecDeque::new(),
                total_bytes: 0,
                max_bytes: None,
                evictions: 0,
            }),
        }
    }
}

impl EvalCache {
    /// Creates a cache bounded to an explicit `max_bytes` of estimated device memory.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                order: VecDeque::new(),
                total_bytes: 0,
                max_bytes: Some(max_bytes),
                evictions: 0,
            }),
        }
    }

    /// Active byte budget (resolved lazily from `client` when created with [`Self::default`]).
    pub fn budget_for(&self, client: &Client) -> usize {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *inner.max_bytes.get_or_insert_with(|| lucem_core::memory::device_budget_bytes(client))
    }

    /// Looks up `key`, promoting it to most-recently-used on hit.
    pub fn get(&self, key: &str) -> Option<Value> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let v = inner.entries.get(key).map(|(v, _)| v.clone())?;
        if let Some(pos) = inner.order.iter().position(|k| k == key) {
            inner.order.remove(pos);
            inner.order.push_back(key.to_string());
        }
        Some(v)
    }

    /// Inserts `value` under `key`, evicting least-recently-used entries past the byte budget
    /// OR when CubeCL's real `bytes_reserved` exceeds 3/4 of device memory (calling
    /// `client.memory_cleanup()` so freed pages return to the driver).
    pub fn put(&self, client: &Client, key: String, value: Value) {
        let bytes = value_bytes(&value);
        let limit = self.budget_for(client);
        if bytes > limit {
            return;
        }
        // Check real CubeCL pool reservation first: returning freed pages often makes room without evicting.
        let mut real_over = lucem_core::memory::device_reserved_ceiling(client, limit)
            .is_some_and(|(reserved, most)| reserved.saturating_add(bytes as u64) > most);
        if real_over {
            let _ = client.memory_cleanup();
            real_over = lucem_core::memory::device_reserved_ceiling(client, limit)
                .is_some_and(|(reserved, most)| reserved.saturating_add(bytes as u64) > most);
        }
        let evicted_any = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, old_bytes)) = inner.entries.remove(&key) {
                inner.total_bytes = inner.total_bytes.saturating_sub(old_bytes);
                if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                    inner.order.remove(pos);
                }
            }
            let mut freed = false;
            while inner.total_bytes.saturating_add(bytes) > limit || (real_over && !inner.order.is_empty()) {
                let Some(oldest) = inner.order.pop_front() else { break };
                if let Some((_, b)) = inner.entries.remove(&oldest) {
                    inner.total_bytes = inner.total_bytes.saturating_sub(b);
                    inner.evictions += 1;
                    freed = true;
                    if real_over && inner.total_bytes.saturating_add(bytes) <= limit {
                        // Drop lock-free cleanup after loop; evict at least `bytes` worth when real_over
                        if b >= bytes {
                            real_over = false;
                        }
                    }
                }
            }
            inner.total_bytes += bytes;
            inner.entries.insert(key.clone(), (value, bytes));
            inner.order.push_back(key);
            freed
        };
        if evicted_any {
            let _ = client.memory_cleanup();
        }
    }

    /// Drops all cached device buffers (e.g. when a device is reset or a batch job needs VRAM).
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.entries.clear();
        inner.order.clear();
        inner.total_bytes = 0;
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated bytes currently held on the device.
    pub fn bytes(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).total_bytes
    }

    /// Total number of entries evicted to stay within budget.
    pub fn evictions(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).evictions
    }
}

impl lucem_core::memory::GpuMemoryHolder for EvalCache {
    fn priority(&self) -> lucem_core::memory::EvictPriority {
        lucem_core::memory::EvictPriority::Onscreen
    }

    fn kept_bytes(&self) -> usize {
        self.inner.try_lock().map_or(0, |g| g.total_bytes)
    }

    fn evict_bytes(&self, target_bytes: usize) -> usize {
        let Ok(mut inner) = self.inner.try_lock() else { return 0 };
        let mut freed = 0usize;
        while freed < target_bytes {
            let Some(oldest) = inner.order.pop_front() else { break };
            if let Some((_, b)) = inner.entries.remove(&oldest) {
                inner.total_bytes = inner.total_bytes.saturating_sub(b);
                inner.evictions += 1;
                freed += b;
            }
        }
        freed
    }
}

/// Computes the deterministic structural cache key of node `id` for `region_tag`.
fn node_cache_key(plan: &Plan, id: &str, region_tag: &str, keys: &mut HashMap<String, String>, depth: usize) -> Result<String, String> {
    if let Some(k) = keys.get(id) {
        return Ok(k.clone());
    }
    if depth > plan.len() + 1 {
        return Err("the graph has a loop".into());
    }
    let node = plan.get(id).ok_or_else(|| format!("no node {id:?}"))?;
    let key = match &node.op {
        None => format!("{region_tag}|src:{id}"),
        Some(op) => {
            if !node.enabled {
                let first = node.inputs.first().cloned().flatten().ok_or_else(|| format!("{} ({id}) is off and has no input", op.title()))?;
                node_cache_key(plan, &first, region_tag, keys, depth + 1)?
            } else {
                let ports = op.inputs();
                let mut in_keys = Vec::with_capacity(ports.len());
                for (i, (_, name)) in ports.iter().enumerate() {
                    let from = node.inputs.get(i).cloned().flatten().ok_or_else(|| format!("{} ({id}): its {name} input is not connected", op.title()))?;
                    in_keys.push(node_cache_key(plan, &from, region_tag, keys, depth + 1)?);
                }
                let op_json = serde_json::to_string(op).unwrap_or_else(|_| op.tag().to_string());
                format!("{region_tag}|op:{op_json}|in:[{}]", in_keys.join(","))
            }
        }
    };
    keys.insert(id.to_string(), key.clone());
    Ok(key)
}

/// Walks backwards from `id` collecting a maximal chain of consecutive uncached, enabled unary
/// pointwise ops (`Image -> Image`). Returns `(base_input_id, ops_in_execution_order, node_ids)`.
fn collect_pointwise_chain(
    plan: &Plan,
    id: &str,
    cache: Option<&EvalCache>,
    memo: &HashMap<String, Value>,
    keys: &HashMap<String, String>,
) -> Option<(String, Vec<PointwiseOp>, Vec<String>)> {
    let mut ops_rev = Vec::new();
    let mut ids_rev = Vec::new();
    let mut cur = id.to_string();
    loop {
        if memo.contains_key(&cur) {
            break;
        }
        if let Some(c) = cache
            && let Some(k) = keys.get(&cur)
            && c.get(k).is_some()
        {
            break;
        }
        let node = plan.get(&cur)?;
        let Some(op) = &node.op else { break };
        if !node.enabled {
            // Skip through disabled passthrough node
            let prev = node.inputs.first().cloned().flatten()?;
            cur = prev;
            continue;
        }
        let Some(pw) = op.pointwise_op() else { break };
        let prev = node.inputs.first().cloned().flatten()?;
        ops_rev.push(pw);
        ids_rev.push(cur);
        cur = prev;
    }
    if ops_rev.len() >= 2 {
        ops_rev.reverse();
        ids_rev.reverse();
        Some((cur, ops_rev, ids_rev))
    } else {
        None
    }
}

/// The value of `target`; `source(id)` gives each source node's value.
pub fn evaluate(client: &Client, plan: &Plan, target: &str, source: &mut dyn FnMut(&str) -> Result<Value, String>) -> Result<Value, String> {
    let mut memo: HashMap<String, Value> = HashMap::new();
    let mut timings = EvalTimings::default();
    eval_inner(client, plan, target, source, &mut memo, None, None, &mut timings, 0)
}

/// Evaluates `target` using `cache` for incremental device-resident reuse, pointwise run fusion,
/// pinned autotune choices, and per-step timing telemetry.
pub fn evaluate_cached(
    client: &Client,
    plan: &Plan,
    target: &str,
    region_tag: &str,
    cache: &EvalCache,
    source: &mut dyn FnMut(&str) -> Result<Value, String>,
) -> Result<(Value, EvalTimings), String> {
    let _pinned = lucem_core::compute::pin_tuned_choices();
    let mut keys = HashMap::new();
    let _ = node_cache_key(plan, target, region_tag, &mut keys, 0)?;
    let mut memo: HashMap<String, Value> = HashMap::new();
    let mut timings = EvalTimings::default();
    let val = eval_inner(client, plan, target, source, &mut memo, Some(cache), Some(&keys), &mut timings, 0)?;
    timings.kept_bytes = cache.bytes();
    timings.budget_bytes = cache.budget_for(client);
    timings.reserved_bytes = lucem_core::memory::device_reserved_ceiling(client, timings.budget_bytes).map_or(0, |(r, _)| r);
    timings.evictions = cache.evictions();
    Ok((val, timings))
}

#[allow(clippy::too_many_arguments)]
fn eval_inner(
    client: &Client,
    plan: &Plan,
    id: &str,
    source: &mut dyn FnMut(&str) -> Result<Value, String>,
    memo: &mut HashMap<String, Value>,
    cache: Option<&EvalCache>,
    keys: Option<&HashMap<String, String>>,
    timings: &mut EvalTimings,
    depth: usize,
) -> Result<Value, String> {
    if let Some(v) = memo.get(id) {
        return Ok(v.clone());
    }
    if depth > plan.len() + 1 {
        return Err("the graph has a loop".into());
    }
    let node = plan.get(id).ok_or_else(|| format!("no node {id:?}"))?;

    // Check cross-evaluation device cache before computing
    if let (Some(cache), Some(keys)) = (cache, keys)
        && let Some(key) = keys.get(id)
        && let Some(hit) = cache.get(key)
    {
        timings.cache_hits += 1;
        if node.op.is_none() {
            timings.input_reused = true;
        }
        memo.insert(id.to_string(), hit.clone());
        return Ok(hit);
    }

    // Check if this node terminates an uncached run of >= 2 consecutive unary pointwise ops
    if let Some(keys_ref) = keys
        && let Some((base_id, pw_ops, chain_ids)) = collect_pointwise_chain(plan, id, cache, memo, keys_ref)
    {
        // Validate each op in the fused chain first so invalid params still report clearly
        for cid in &chain_ids {
            if let Some(Some(op)) = plan.get(cid).map(|n| &n.op) {
                op.validate()?;
            }
        }
        let base_val = eval_inner(client, plan, &base_id, source, memo, cache, keys, timings, depth + 1)?;
        let Value::Image(base_img) = base_val else {
            return Err(format!("pointwise chain ({id}): input {base_id} must be an image, got {}", base_val.port().name()));
        };
        let t = Instant::now();
        let run = PointwiseRun::upload::<f32>(client, &pw_ops);
        let out = run.apply::<f32>(client, &base_img).map_err(|e| e.to_string())?;
        if measuring() {
            #[cfg(not(target_family = "wasm"))]
            let _ = lucem_core::cubecl::future::block_on(client.sync());
            let dt = ms(t);
            timings.steps.push((format!("fused_pointwise({})", pw_ops.len()), dt));
            timings.compute_ms += dt;
        } else {
            timings.compute_ms += ms(t);
        }
        timings.fused_runs += 1;
        let value = Value::Image(out);
        if let Some(c) = cache
            && let Some(key) = keys_ref.get(id)
        {
            c.put(client, key.clone(), value.clone());
        }
        memo.insert(id.to_string(), value.clone());
        return Ok(value);
    }

    let value = match &node.op {
        None => {
            let t = Instant::now();
            let v = source(id)?;
            let dt = ms(t);
            timings.source_ms += dt;
            if measuring() {
                timings.steps.push((format!("source:{id}"), dt));
            }
            v
        }
        Some(op) => {
            let ports = op.inputs();
            let mut inputs = Vec::with_capacity(ports.len());
            for (i, (want, name)) in ports.iter().enumerate() {
                let from = node.inputs.get(i).cloned().flatten().ok_or_else(|| format!("{} ({id}): its {name} input is not connected", op.title()))?;
                let v = eval_inner(client, plan, &from, source, memo, cache, keys, timings, depth + 1)?;
                if v.port() != *want {
                    return Err(format!("{} ({id}): its {name} input takes {}, {from} gives {}", op.title(), want.name(), v.port().name()));
                }
                inputs.push(v);
            }
            if !node.enabled {
                match inputs.into_iter().next() {
                    Some(v) if v.port() == op.output() => v,
                    _ => return Err(format!("{} ({id}) is off, and its input is not a {}: turn it on or remove it", op.title(), op.output().name())),
                }
            } else {
                // Inputs of one op must be the same size (a region of one image)
                let dims: Vec<(usize, usize)> = inputs.iter().filter_map(Value::dims).collect();
                if dims.windows(2).any(|w| w[0] != w[1]) {
                    return Err(format!("{} ({id}): its inputs differ in size", op.title()));
                }
                let t = Instant::now();
                let v = op.run(client, &inputs)?;
                if measuring() {
                    #[cfg(not(target_family = "wasm"))]
                    let _ = lucem_core::cubecl::future::block_on(client.sync());
                    let dt = ms(t);
                    timings.steps.push((format!("{}:{id}", op.tag()), dt));
                    timings.compute_ms += dt;
                } else {
                    timings.compute_ms += ms(t);
                }
                v
            }
        }
    };

    if let (Some(cache), Some(keys)) = (cache, keys)
        && let Some(key) = keys.get(id)
    {
        cache.put(client, key.clone(), value.clone());
    }
    memo.insert(id.to_string(), value.clone());
    Ok(value)
}

/// Crops the halo-padded 2-D raster `value` (spanning `rx0..rx0+rw`, `ry0..ry0+rh`) on the device
/// to the visible rectangle `[vx0..vx0+vw, vy0..vy0+vh]` and downloads only those `vw × vh` pixels.
///
/// Returns `(pixels, readback_ms)`.
#[allow(clippy::too_many_arguments)]
pub fn crop_and_download(
    client: &Client,
    value: &Value,
    rx0: usize,
    ry0: usize,
    rw: usize,
    rh: usize,
    vx0: usize,
    vy0: usize,
    vw: usize,
    vh: usize,
) -> Result<(Vec<f32>, f64), String> {
    let t = Instant::now();
    let ox = vx0.checked_sub(rx0).ok_or_else(|| format!("crop x0 ({vx0}) < region x0 ({rx0})"))?;
    let oy = vy0.checked_sub(ry0).ok_or_else(|| format!("crop y0 ({vy0}) < region y0 ({ry0})"))?;
    let right = rw.checked_sub(ox + vw).ok_or_else(|| format!("crop x1 ({}) exceeds region width {rw}", ox + vw))?;
    let bottom = rh.checked_sub(oy + vh).ok_or_else(|| format!("crop y1 ({}) exceeds region height {rh}", oy + vh))?;
    let no_crop = ox == 0 && oy == 0 && right == 0 && bottom == 0;
    let pixels = match value {
        Value::Image(a) => {
            if no_crop {
                a.to_host(client)
            } else {
                let cropped = lucem_ops::intensity::convert::crop(client, a, &[(oy, bottom), (ox, right)]).map_err(|e| e.to_string())?;
                cropped.to_host(client)
            }
        }
        Value::Mask(a) | Value::Labels(a) => {
            let raw = if no_crop {
                a.to_host(client)
            } else {
                let cropped = lucem_ops::intensity::convert::crop(client, a, &[(oy, bottom), (ox, right)]).map_err(|e| e.to_string())?;
                cropped.to_host(client)
            };
            raw.into_iter().map(|v| v as f32).collect()
        }
        Value::Table(_) => return Err("a table is not a raster".into()),
    };
    Ok((pixels, ms(t)))
}

/// Every node of `targets` (and what they need), each computed once; a node that fails gives its
/// error and the others go on.
pub fn evaluate_all(client: &Client, plan: &Plan, targets: &[String], source: &mut dyn FnMut(&str) -> Result<Value, String>) -> BTreeMap<String, Result<Value, String>> {
    let mut memo: HashMap<String, Value> = HashMap::new();
    let mut timings = EvalTimings::default();
    let mut out = BTreeMap::new();
    for t in targets {
        let r = eval_inner(client, plan, t, source, &mut memo, None, None, &mut timings, 0);
        out.insert(t.clone(), r);
    }
    out
}

/// The plan with every op set for a level of downsample `ds` (see [`Op::at_downsample`]).
pub fn at_downsample(plan: &Plan, ds: f64) -> Plan {
    plan.iter()
        .map(|(id, n)| (id.clone(), PlanNode { op: n.op.as_ref().map(|o| o.at_downsample(ds)), enabled: n.enabled, inputs: n.inputs.clone() }))
        .collect()
}

/// Pixels of context `target` needs around a region: the largest sum of halos on any path to a
/// source.
pub fn halo(plan: &Plan, target: &str) -> usize {
    fn walk(plan: &Plan, id: &str, depth: usize) -> usize {
        if depth > plan.len() {
            return 0;
        }
        let Some(n) = plan.get(id) else { return 0 };
        let own = match (&n.op, n.enabled) {
            (Some(op), true) => op.halo(),
            _ => 0,
        };
        own + n.inputs.iter().flatten().map(|i| walk(plan, i, depth + 1)).max().unwrap_or(0)
    }
    walk(plan, target, 0)
}

/// What `id` gives (sources: `source_type`).
pub fn output_type(plan: &Plan, id: &str, source_type: &dyn Fn(&str) -> Option<PortType>) -> Option<PortType> {
    let n = plan.get(id)?;
    match &n.op {
        None => source_type(id),
        Some(op) => Some(op.output()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_region_rounds_outward_and_clamps() {
        assert_eq!(align_region(10, 20, 100, 130, 500, 500), (0, 0, 128, 192));
        assert_eq!(align_region(64, 128, 192, 256, 500, 500), (64, 128, 128, 128));
        assert_eq!(align_region(450, 450, 500, 500, 500, 500), (448, 448, 52, 52));
    }
}
