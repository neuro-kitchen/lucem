//! Device (GPU) and host (RAM) memory budgets, size-class rounding, and priority eviction.
//!
//! Lessons from `dsp-kitchen`:
//! 1. **Count bytes, not objects.** Every cache enforces a byte budget before allocating.
//! 2. **Size from the hardware, lazily.** GPU budget defaults to half of
//!    `client.properties().memory.max_memory()` (or `LUCEM_DEVICE_MEMORY` /
//!    `LUCEM_DEVICE_MEMORY_LIMIT`, with a conservative browser fallback on wasm). RAM budget
//!    defaults to a quarter of system RAM (or `LUCEM_RAM_MEMORY` / `[compute] memory`).
//! 3. **Evict before allocating & call `client.memory_cleanup()`.** Holders register with the
//!    global GPU budget, are evicted in priority order (off-screen scratch -> off-screen results ->
//!    on-screen LRU, skipping busy holders, using `try_lock` only so two holders never deadlock),
//!    and CubeCL returns freed pages to the driver via `client.memory_cleanup()`.
//! 4. **Size-class rounding.** [`size_class`] rounds variable allocation sizes to
//!    `2^k × {1, 1.25, 1.5, 1.75}` (≤ 25% overhead) so CubeCL's pool reuses freed pages.

use std::sync::{Arc, Mutex, Once, Weak};

use cubecl::prelude::*;
use cubecl::{MemoryScope, MemoryUsage};

use crate::error::{CoreError, CoreResult};

/// Environment variable: device memory limit, e.g. `7GiB`, `6500MB`, `7516192768`.
pub const DEVICE_MEMORY_LIMIT_ENV: &str = "LUCEM_DEVICE_MEMORY_LIMIT";
/// Short alias matching `DSPK_DEVICE_MEMORY`.
pub const DEVICE_MEMORY_ENV: &str = "LUCEM_DEVICE_MEMORY";
/// Environment variable: host RAM cache limit, e.g. `8GiB`, `2048MiB`.
pub const RAM_MEMORY_ENV: &str = "LUCEM_RAM_MEMORY";

/// Allocations below this size skip the blocking CubeCL device memory report query.
pub const CHECKED_ALLOCATION_BYTES: u64 = 64 << 20;

/// Conservative GPU budget fallback on wasm/WebGPU (where `max_memory` is unreported).
pub const BROWSER_GPU_BUDGET_BYTES: usize = 256 * 1024 * 1024;
/// Default native GPU budget fallback when the runtime reports 0 bytes.
pub const DEFAULT_GPU_BUDGET_BYTES: usize = 512 * 1024 * 1024;
/// Default RAM cache budget when system RAM cannot be queried (e.g. wasm).
pub const DEFAULT_RAM_BUDGET_BYTES: usize = 512 * 1024 * 1024;

static LIMIT: Mutex<Option<u64>> = Mutex::new(None);
static RAM_LIMIT: Mutex<Option<usize>> = Mutex::new(None);
static FROM_ENV: Once = Once::new();
static HOLDERS: Mutex<Vec<Weak<dyn GpuMemoryHolder>>> = Mutex::new(Vec::new());

/// Eviction priority tier for GPU memory holders (lower = evicted first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvictPriority {
    /// Off-screen scratch / workspace buffers (cheapest to drop).
    OffscreenScratch = 0,
    /// Off-screen cached results.
    OffscreenResult = 1,
    /// On-screen active caches (evicted LRU only when off-screen is exhausted).
    Onscreen = 2,
}

/// A GPU memory holder registered with the global budget.
///
/// Implementations MUST use non-blocking `try_lock` inside [`evict_bytes`] so concurrent callers
/// making room never deadlock. Busy holders simply return `0`.
pub trait GpuMemoryHolder: Send + Sync {
    /// Current eviction priority tier.
    fn priority(&self) -> EvictPriority;
    /// Estimated device bytes currently held.
    fn kept_bytes(&self) -> usize;
    /// Frees at least `target_bytes` (or as much as possible) using `try_lock`. Returns freed bytes.
    fn evict_bytes(&self, target_bytes: usize) -> usize;
}

/// Registers a GPU memory holder with the process-wide budget.
pub fn register_gpu_holder(holder: &Arc<dyn GpuMemoryHolder>) {
    let mut list = HOLDERS.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|w| w.strong_count() > 0);
    list.push(Arc::downgrade(holder));
}

/// Total bytes tracked across all registered GPU holders.
pub fn tracked_gpu_bytes() -> usize {
    let mut list = HOLDERS.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|w| w.strong_count() > 0);
    list.iter().filter_map(Weak::upgrade).map(|h| h.kept_bytes()).sum()
}

/// What CubeCL's device pools currently reserve (`bytes_reserved`) and the 3/4-of-device ceiling
/// (`None` on wasm where `memory_report` is not used for eviction).
pub fn device_reserved_ceiling(client: &Client, budget: usize) -> Option<(u64, u64)> {
    #[cfg(not(target_family = "wasm"))]
    {
        let reserved = memory_usage(client).bytes_reserved;
        let total = client.properties().memory.max_memory().unwrap_or((budget as u64).saturating_mul(2));
        Some((reserved, total / 4 * 3))
    }
    #[cfg(target_family = "wasm")]
    {
        let _ = (client, budget);
        None
    }
}

/// Evicts registered holders in priority order until BOTH `tracked_gpu_bytes() + need <= budget`
/// AND CubeCL's real `bytes_reserved + need <= 3/4 * max_memory` (calling `client.memory_cleanup()`
/// first so freed-but-reserved pages return to the driver before evicting live caches).
///
/// Never holds the global registry lock while calling into a holder. Returns total bytes freed.
pub fn make_gpu_room(client: &Client, need: usize, budget: usize) -> usize {
    let real_over = || device_reserved_ceiling(client, budget).is_some_and(|(reserved, most)| reserved.saturating_add(need as u64) > most);
    let mut is_real_over = real_over();
    if is_real_over {
        // Freed-but-reserved pages go back to the driver first: often room enough on its own.
        let _ = client.memory_cleanup();
        is_real_over = real_over();
    }
    let snapshot: Vec<Arc<dyn GpuMemoryHolder>> = {
        let mut list = HOLDERS.lock().unwrap_or_else(|e| e.into_inner());
        list.retain(|w| w.strong_count() > 0);
        list.iter().filter_map(Weak::upgrade).collect()
    };
    let current: usize = snapshot.iter().map(|h| h.kept_bytes()).sum();
    if !is_real_over && current.saturating_add(need) <= budget {
        return 0;
    }
    let mut to_free = current.saturating_add(need).saturating_sub(budget);
    let mut ordered = snapshot;
    ordered.sort_by_key(|h| h.priority());
    let mut freed_total = 0usize;
    for holder in ordered {
        if !is_real_over && to_free == 0 {
            break;
        }
        let ask = to_free.max(need);
        let freed = holder.evict_bytes(ask);
        freed_total += freed;
        to_free = to_free.saturating_sub(freed);
        if is_real_over && freed > 0 {
            let _ = client.memory_cleanup();
            is_real_over = real_over();
        }
    }
    if freed_total > 0 {
        let _ = client.memory_cleanup();
    }
    freed_total
}

/// Rounds `n` up to a size class `2^k × {1, 1.25, 1.5, 1.75}` (at most 25% overhead) so CubeCL's
/// allocator pool reuses freed pages across nearby variable buffer sizes.
pub fn size_class(n: usize) -> usize {
    if n <= 64 {
        return n.max(1);
    }
    let pow = 1usize << (usize::BITS - 1 - n.leading_zeros());
    let step = pow / 4;
    if n <= pow {
        pow
    } else if n <= pow + step {
        pow + step
    } else if n <= pow + 2 * step {
        pow + 2 * step
    } else if n <= pow + 3 * step {
        pow + 3 * step
    } else {
        pow.saturating_mul(2)
    }
}

/// Lazy hardware-sized GPU budget in bytes:
/// - Explicit `LUCEM_DEVICE_MEMORY` / `LUCEM_DEVICE_MEMORY_LIMIT` or [`set_device_memory_limit`].
/// - Otherwise half of `client.properties().memory.max_memory()` on native.
/// - Conservative [`BROWSER_GPU_BUDGET_BYTES`] on wasm or when the driver reports 0.
pub fn device_budget_bytes(client: &Client) -> usize {
    if let Some(limit) = device_memory_limit() {
        return (limit as usize).max(16 * 1024 * 1024);
    }
    let memory = &client.properties().memory;
    let bytes = match memory.max_memory() {
        Some(total) => (total / 2).max(64 << 20),
        None => (4 * memory.max_page_size).clamp(BROWSER_GPU_BUDGET_BYTES as u64, 2 << 30),
    };
    usize::try_from(bytes).unwrap_or(DEFAULT_GPU_BUDGET_BYTES)
}

/// Sets the device memory limit in bytes (`None`: hardware rule). Overrides env vars.
pub fn set_device_memory_limit(bytes: Option<u64>) {
    read_env();
    *LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = bytes;
}

/// The explicit device memory limit in bytes, if one is set.
pub fn device_memory_limit() -> Option<u64> {
    read_env();
    *LIMIT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets the process-wide host RAM cache limit in bytes (`None`: automatic = quarter of RAM).
pub fn set_ram_memory_limit(bytes: Option<usize>) {
    read_env();
    *RAM_LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = bytes;
}

/// Current process-wide host RAM cache limit in bytes.
pub fn ram_memory_limit() -> usize {
    read_env();
    if let Some(limit) = *RAM_LIMIT.lock().unwrap_or_else(|e| e.into_inner()) {
        return limit.max(32 * 1024 * 1024);
    }
    detect_automatic_ram_budget()
}

fn detect_automatic_ram_budget() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) {
                        let quarter = (kb * 1024 / 4) as usize;
                        return quarter.clamp(128 * 1024 * 1024, 16 * 1024 * 1024 * 1024);
                    }
                }
            }
        }
    }
    DEFAULT_RAM_BUDGET_BYTES
}

/// What CubeCL's memory pools hold on `client`'s device (all streams).
pub fn memory_usage(client: &Client) -> MemoryUsage {
    client.memory_report(MemoryScope::Device).usage()
}

/// `Ok` if `bytes` more fit the limit next to what the device's pools already use; a no-op without a
/// limit or below [`CHECKED_ALLOCATION_BYTES`].
pub fn check_allocation(client: &Client, what: &str, bytes: u64) -> CoreResult<()> {
    let Some(limit) = device_memory_limit() else { return Ok(()) };
    if bytes < CHECKED_ALLOCATION_BYTES {
        return Ok(());
    }
    let in_use = memory_usage(client).bytes_in_use;
    let available = limit.saturating_sub(in_use);
    if bytes <= available {
        Ok(())
    } else {
        Err(CoreError::OutOfDeviceMemory { what: what.to_string(), requested: bytes, available, limit })
    }
}

fn read_env() {
    FROM_ENV.call_once(|| {
        let dev = std::env::var(DEVICE_MEMORY_ENV)
            .ok()
            .or_else(|| std::env::var(DEVICE_MEMORY_LIMIT_ENV).ok())
            .and_then(|v| parse_bytes(&v));
        if let Some(bytes) = dev {
            *LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = Some(bytes);
        }
        if let Some(ram) = std::env::var(RAM_MEMORY_ENV).ok().and_then(|v| parse_bytes(&v)) {
            *RAM_LIMIT.lock().unwrap_or_else(|e| e.into_inner()) = Some(ram as usize);
        }
    });
}

/// `"7GiB"`, `"6.5 GB"`, `"512MiB"`, `"1024"` → bytes (decimal units for kB/MB/GB, binary for
/// KiB/MiB/GiB). `None` for anything else.
pub fn parse_bytes(text: &str) -> Option<u64> {
    let text = text.trim();
    let split = text.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let value: f64 = number.parse().ok()?;
    let scale: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "kb" => 1e3,
        "mb" => 1e6,
        "gb" => 1e9,
        "kib" => 1024.0,
        "mib" => 1024.0 * 1024.0,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    (value >= 0.0).then_some((value * scale) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_bytes("1024"), Some(1024));
        assert_eq!(parse_bytes("7GiB"), Some(7 << 30));
        assert_eq!(parse_bytes(" 6.5 GB "), Some(6_500_000_000));
        assert_eq!(parse_bytes("512mib"), Some(512 << 20));
        assert_eq!(parse_bytes("lots"), None);
        assert_eq!(parse_bytes("5 TB"), None);
    }

    #[test]
    fn size_class_bounds_overhead_to_25_percent() {
        for n in [1, 64, 65, 100, 1000, 1024, 1025, 1280, 1536, 1792, 2048, 100_000, 1_048_577] {
            let c = size_class(n);
            assert!(c >= n, "size_class({n}) = {c} < {n}");
            assert!(c <= n + n / 4 + 64, "size_class({n}) = {c} exceeds 25% overhead");
        }
    }
}
