//! Pure-Rust OME-Zarr v0.4 (`zarr_format: 2`) container reader with Blosc-LZ4 + byte-unshuffle decoding.
//!
//! Treats an `.ome.zarr` directory as a single unified container:
//! - Parses `.zattrs` (`multiscales` pyramid levels and `omero` channel metadata) and `labels/.zattrs`
//! - Resolves any path inside an `.ome.zarr` bundle back to its `.ome.zarr` container root
//! - Allows switching pyramid resolution levels (`s0`, `s1`, `s2`, `s3`), individual or composite
//!   channels (`DAPI_deblur`, `RFP`), and label groups (`cells`, `nuclei`) on the fly in the Studio.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{IoError, IoResult};
use crate::image_io::LoadedImage;

/// Metadata for an opened `.ome.zarr` container attached to [`LoadedImage`].
#[derive(Debug, Clone)]
pub struct OmeZarrContainerInfo {
    /// Path to the `.ome.zarr` root directory on disk.
    pub root_path: PathBuf,
    /// Display name of the `.ome.zarr` container (e.g. `"TDTNI1_P4_M_SAL_S1_stack.ome.zarr"`).
    pub container_name: String,
    /// Available multiscale pyramid levels from `.zattrs` (e.g. `["s0", "s1", "s2", "s3"]`).
    pub pyramid_levels: Vec<String>,
    /// `(level_name, width, height)` for each multiscale pyramid level.
    pub level_dimensions: Vec<(String, usize, usize)>,
    /// Currently loaded pyramid level (e.g. `"s2"`).
    pub active_level: String,
    /// Whether zooming in/out automatically switches the active pyramid level (`s3 ↔ s2 ↔ s1 ↔ s0`).
    pub auto_pyramid: bool,
    /// Channel labels from `.zattrs` `omero.channels` (e.g. `["DAPI_deblur", "RFP"]`).
    pub channel_labels: Vec<String>,
    /// `None` = composite across channels; `Some(c)` = single channel `c`.
    pub active_channel: Option<usize>,
    /// Available segmentation label groups from `labels/.zattrs` (e.g. `["cells", "nuclei"]`).
    pub label_groups: Vec<String>,
    /// `None` = primary image pyramid; `Some(name)` = `labels/<name>/<active_level>`.
    pub active_label_group: Option<String>,
}

impl OmeZarrContainerInfo {
    /// Chooses the appropriate pyramid level (`s3..s0`) for a given on-screen `displayed_width`
    /// in viewport pixels, with hysteresis so scrolling near a threshold does not oscillate.
    pub fn select_level_for_display_width(&self, displayed_width: f32) -> String {
        if self.level_dimensions.is_empty() {
            return self.active_level.clone();
        }
        // Sort levels from coarsest (smallest width) to finest (largest width)
        let mut sorted = self.level_dimensions.clone();
        sorted.sort_by_key(|&(_, w, _)| w);

        // Find the coarsest level where `level_width * 1.15 >= displayed_width`
        let mut target_idx = sorted.len() - 1;
        for (i, &(_, w, _)) in sorted.iter().enumerate() {
            if (w as f32) * 1.15 >= displayed_width {
                target_idx = i;
                break;
            }
        }

        // Hysteresis: if currently on a finer level `curr_idx > target_idx`, only step down if
        // `displayed_width < sorted[target_idx].width * 0.90`.
        if let Some(curr_idx) = sorted.iter().position(|(l, _, _)| *l == self.active_level) {
            if curr_idx > target_idx {
                let step_down_w = sorted[curr_idx - 1].1 as f32;
                if displayed_width >= step_down_w * 0.90 {
                    return self.active_level.clone();
                }
            }
        }

        sorted[target_idx].0.clone()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ZarrCompressor {
    id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ZArrayMeta {
    shape: Vec<usize>,
    chunks: Vec<usize>,
    dtype: String,
    #[serde(default = "default_dim_sep")]
    dimension_separator: String,
    compressor: Option<ZarrCompressor>,
}

fn default_dim_sep() -> String {
    "/".to_string()
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RootZAttrs {
    #[serde(default)]
    multiscales: Vec<MultiscaleEntry>,
    #[serde(default)]
    omero: Option<OmeroMeta>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct MultiscaleEntry {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    datasets: Vec<MultiscaleDataset>,
}

#[derive(Debug, Clone, Deserialize)]
struct MultiscaleDataset {
    path: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct OmeroMeta {
    #[serde(default)]
    channels: Vec<OmeroChannel>,
}

#[derive(Debug, Clone, Deserialize)]
struct OmeroChannel {
    #[serde(default)]
    label: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct LabelsZAttrs {
    #[serde(default)]
    labels: Vec<String>,
}

/// Decompresses a raw LZ4 block (`src`) of known `uncompressed_len`.
fn lz4_decompress_block(src: &[u8], uncompressed_len: usize) -> IoResult<Vec<u8>> {
    let mut out = vec![0u8; uncompressed_len];
    let mut ip = 0usize;
    let mut op = 0usize;
    let n = src.len();

    while ip < n && op < uncompressed_len {
        let token = src[ip];
        ip += 1;

        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            while ip < n {
                let s = src[ip] as usize;
                ip += 1;
                lit_len += s;
                if s != 255 {
                    break;
                }
            }
        }

        if ip + lit_len > n || op + lit_len > uncompressed_len {
            return Err(IoError::OmeZarr("LZ4 literal overrun".to_string()));
        }
        out[op..op + lit_len].copy_from_slice(&src[ip..ip + lit_len]);
        ip += lit_len;
        op += lit_len;

        if ip >= n || op >= uncompressed_len {
            break;
        }

        if ip + 2 > n {
            return Err(IoError::OmeZarr("LZ4 truncated match offset".to_string()));
        }
        let offset = (src[ip] as usize) | ((src[ip + 1] as usize) << 8);
        ip += 2;
        if offset == 0 || offset > op {
            return Err(IoError::OmeZarr(format!("LZ4 invalid match offset {offset} at op={op}")));
        }

        let mut match_len = ((token & 0x0F) as usize) + 4;
        if match_len == 19 {
            while ip < n {
                let s = src[ip] as usize;
                ip += 1;
                match_len += s;
                if s != 255 {
                    break;
                }
            }
        }

        if op + match_len > uncompressed_len {
            return Err(IoError::OmeZarr("LZ4 match overrun".to_string()));
        }

        let mut match_pos = op - offset;
        for _ in 0..match_len {
            out[op] = out[match_pos];
            op += 1;
            match_pos += 1;
        }
    }

    if op != uncompressed_len {
        return Err(IoError::OmeZarr(format!(
            "LZ4 decompressed length mismatch: got {op}, expected {uncompressed_len}"
        )));
    }
    Ok(out)
}

/// Decompresses a Blosc v1 frame (`bytes`) using LZ4 and byte-unshuffle when flagged.
fn blosc_decompress(bytes: &[u8]) -> IoResult<Vec<u8>> {
    if bytes.len() < 16 {
        return Err(IoError::OmeZarr("Blosc header shorter than 16 bytes".to_string()));
    }
    let flags = bytes[2];
    let typesize = (bytes[3] as usize).max(1);
    let nbytes = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let blocksize = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let cbytes = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;

    if cbytes > bytes.len() {
        return Err(IoError::OmeZarr("Blosc cbytes exceeds buffer length".to_string()));
    }

    let is_memcpy = (flags & 0x02) != 0;
    if is_memcpy {
        if bytes.len() < 16 + nbytes {
            return Err(IoError::OmeZarr("Blosc memcpy buffer truncated".to_string()));
        }
        return Ok(bytes[16..16 + nbytes].to_vec());
    }

    let do_shuffle = (flags & 0x01) != 0;
    let bsize_default = blocksize.max(1);
    let num_blocks = (nbytes + bsize_default - 1) / bsize_default;

    if bytes.len() < 16 + 4 * num_blocks {
        return Err(IoError::OmeZarr("Blosc bstarts table truncated".to_string()));
    }

    let mut out = vec![0u8; nbytes];

    for b_idx in 0..num_blocks {
        let bstart_off = 16 + 4 * b_idx;
        let mut off = u32::from_le_bytes([
            bytes[bstart_off],
            bytes[bstart_off + 1],
            bytes[bstart_off + 2],
            bytes[bstart_off + 3],
        ]) as usize;

        let next_block_off = if b_idx + 1 < num_blocks {
            let nb_off = 16 + 4 * (b_idx + 1);
            u32::from_le_bytes([
                bytes[nb_off],
                bytes[nb_off + 1],
                bytes[nb_off + 2],
                bytes[nb_off + 3],
            ]) as usize
        } else {
            cbytes
        };

        let out_block_start = b_idx * bsize_default;
        let bsize = (nbytes - out_block_start).min(bsize_default);

        if off + 4 > bytes.len() {
            return Err(IoError::OmeZarr("Blosc stream length header truncated".to_string()));
        }
        let first_csize =
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) as usize;
        let single_stream_end = off + 4 + first_csize;
        let num_streams = if typesize <= 1 || single_stream_end == next_block_off {
            1
        } else {
            typesize
        };
        let stream_uncomp_len = bsize / num_streams;
        let mut block_raw = Vec::with_capacity(bsize);

        for _ in 0..num_streams {
            if off + 4 > bytes.len() {
                return Err(IoError::OmeZarr("Blosc stream length header truncated".to_string()));
            }
            let csize = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) as usize;
            off += 4;
            if csize == 0 {
                block_raw.resize(block_raw.len() + stream_uncomp_len, 0);
            } else if csize == stream_uncomp_len {
                block_raw.extend_from_slice(&bytes[off..off + csize]);
                off += csize;
            } else {
                if off + csize > bytes.len() {
                    return Err(IoError::OmeZarr("Blosc compressed stream overrun".to_string()));
                }
                let dec = lz4_decompress_block(&bytes[off..off + csize], stream_uncomp_len)?;
                block_raw.extend_from_slice(&dec);
                off += csize;
            }
        }

        let dst_slice = &mut out[out_block_start..out_block_start + bsize];
        if do_shuffle && typesize > 1 {
            let nelems = bsize / typesize;
            for elem_i in 0..nelems {
                for byte_j in 0..typesize {
                    dst_slice[elem_i * typesize + byte_j] = block_raw[byte_j * nelems + elem_i];
                }
            }
        } else {
            dst_slice.copy_from_slice(&block_raw[..bsize]);
        }
    }

    Ok(out)
}

/// Walks up `path` ancestors to find the root `.ome.zarr` container directory.
pub fn find_ome_zarr_root(path: &Path) -> Option<PathBuf> {
    for ancestor in path.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".ome.zarr") && ancestor.is_dir() {
                return Some(ancestor.to_path_buf());
            }
        }
    }
    for ancestor in path.ancestors() {
        if ancestor.is_dir() && ancestor.join(".zattrs").is_file() && ancestor.join(".zgroup").is_file() {
            if let Ok(txt) = fs::read_to_string(ancestor.join(".zattrs")) {
                if txt.contains("multiscales") {
                    return Some(ancestor.to_path_buf());
                }
            }
        }
    }
    None
}

/// Returns `true` if `path` is (or is inside) an `.ome.zarr` container.
pub fn is_ome_zarr_path(path: &Path) -> bool {
    find_ome_zarr_root(path).is_some()
        || path.to_string_lossy().contains(".ome.zarr")
        || path.join(".zarray").is_file()
        || path.join(".zgroup").is_file()
}

/// Parses `.zattrs` and `labels/.zattrs` from the `.ome.zarr` container root.
pub fn inspect_ome_zarr_container(root_dir: &Path) -> IoResult<OmeZarrContainerInfo> {
    let mut container_name = root_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset.ome.zarr")
        .to_string();

    let mut pyramid_levels = Vec::new();
    let mut channel_labels = Vec::new();

    if let Ok(zattrs_str) = fs::read_to_string(root_dir.join(".zattrs")) {
        if let Ok(parsed) = serde_json::from_str::<RootZAttrs>(&zattrs_str) {
            if let Some(ms) = parsed.multiscales.first() {
                if let Some(ref n) = ms.name {
                    if !n.is_empty() {
                        container_name = n.clone();
                    }
                }
                for ds in &ms.datasets {
                    if root_dir.join(&ds.path).join(".zarray").is_file() {
                        pyramid_levels.push(ds.path.clone());
                    }
                }
            }
            if let Some(omero) = parsed.omero {
                for (i, ch) in omero.channels.into_iter().enumerate() {
                    if ch.label.is_empty() {
                        channel_labels.push(format!("Channel {i}"));
                    } else {
                        channel_labels.push(ch.label);
                    }
                }
            }
        }
    }

    if pyramid_levels.is_empty() {
        for cand in ["s0", "s1", "s2", "s3", "0", "1", "2"] {
            if root_dir.join(cand).join(".zarray").is_file() {
                pyramid_levels.push(cand.to_string());
            }
        }
    }

    if pyramid_levels.is_empty() {
        return Err(IoError::OmeZarr(format!(
            "No multiscale pyramid levels found in '{}'",
            root_dir.display()
        )));
    }

    let mut level_dimensions = Vec::new();
    for lvl in &pyramid_levels {
        if let Ok(zstr) = fs::read_to_string(root_dir.join(lvl).join(".zarray")) {
            if let Ok(zmeta) = serde_json::from_str::<ZArrayMeta>(&zstr) {
                let (h, w) = match zmeta.shape.as_slice() {
                    &[h, w] => (h, w),
                    &[_, h, w] => (h, w),
                    &[_, _, h, w] => (h, w),
                    _ => (0, 0),
                };
                if w > 0 && h > 0 {
                    level_dimensions.push((lvl.clone(), w, h));
                }
            }
        }
    }

    // Default to `s2` (or `s1`) when available for instantaneous startup, while allowing automatic
    // or manual switching to `s1` / `s0` when zooming in.
    let active_level = if pyramid_levels.iter().any(|l| l == "s2") {
        "s2".to_string()
    } else if pyramid_levels.iter().any(|l| l == "s1") {
        "s1".to_string()
    } else {
        pyramid_levels[0].clone()
    };

    let mut label_groups = Vec::new();
    let labels_dir = root_dir.join("labels");
    if let Ok(lbl_str) = fs::read_to_string(labels_dir.join(".zattrs")) {
        if let Ok(parsed_lbl) = serde_json::from_str::<LabelsZAttrs>(&lbl_str) {
            for g in parsed_lbl.labels {
                if labels_dir.join(&g).is_dir() {
                    label_groups.push(g);
                }
            }
        }
    }

    Ok(OmeZarrContainerInfo {
        root_path: root_dir.to_path_buf(),
        container_name,
        pyramid_levels,
        level_dimensions,
        active_level,
        auto_pyramid: true,
        channel_labels,
        active_channel: None,
        label_groups,
        active_label_group: None,
    })
}

/// Converts raw decoded chunk bytes of `dtype` into `f32` values.
fn decode_elements_to_f32(raw: &[u8], dtype: &str) -> IoResult<Vec<f32>> {
    match dtype {
        "<u2" | "|u2" => {
            let mut out = Vec::with_capacity(raw.len() / 2);
            for c in raw.chunks_exact(2) {
                out.push(u16::from_le_bytes([c[0], c[1]]) as f32);
            }
            Ok(out)
        }
        "<u1" | "|u1" => Ok(raw.iter().map(|&b| b as f32).collect()),
        "<i4" | "|i4" => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]).max(0) as f32);
            }
            Ok(out)
        }
        "<u4" | "|u4" => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                out.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
            Ok(out)
        }
        "<f4" | "|f4" => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        other => Err(IoError::OmeZarr(format!("Unsupported Zarr dtype '{other}'"))),
    }
}

/// Loads an `.ome.zarr` container (or any path inside it) using its default view settings.
pub fn load_ome_zarr(path: &Path) -> IoResult<LoadedImage> {
    let root = find_ome_zarr_root(path).unwrap_or_else(|| path.to_path_buf());
    let info = inspect_ome_zarr_container(&root)?;
    load_ome_zarr_with_options(&info)
}

/// Loads the specified pyramid level, channel selection, or label group from an [`OmeZarrContainerInfo`].
pub fn load_ome_zarr_with_options(info: &OmeZarrContainerInfo) -> IoResult<LoadedImage> {
    let array_dir = match &info.active_label_group {
        Some(lbl) => {
            let candidate = info.root_path.join("labels").join(lbl).join(&info.active_level);
            if candidate.join(".zarray").is_file() {
                candidate
            } else {
                // Fall back to `s2` or `s0` inside `labels/<lbl>` if `active_level` is not present
                let lbl_root = info.root_path.join("labels").join(lbl);
                ["s1", "s2", "s0"]
                    .iter()
                    .map(|s| lbl_root.join(s))
                    .find(|p| p.join(".zarray").is_file())
                    .unwrap_or(candidate)
            }
        }
        None => info.root_path.join(&info.active_level),
    };

    let zarray_str = fs::read_to_string(array_dir.join(".zarray"))?;
    let meta: ZArrayMeta = serde_json::from_str(&zarray_str)?;

    let (channels, height, width, chunk_c, chunk_y, chunk_x) = match (meta.shape.as_slice(), meta.chunks.as_slice()) {
        (&[h, w], &[cy, cx]) => (1usize, h, w, 1usize, cy, cx),
        (&[c, h, w], &[cc, cy, cx]) => (c, h, w, cc.max(1), cy, cx),
        (&[t, c, h, w], &[_, cc, cy, cx]) if t >= 1 => (c, h, w, cc.max(1), cy, cx),
        _ => {
            return Err(IoError::OmeZarr(format!(
                "Unsupported Zarr shape {:?} / chunks {:?}",
                meta.shape, meta.chunks
            )))
        }
    };

    let n_pixels = height * width;
    let num_c_load = channels.min(3);
    let mut channel_planes = vec![vec![0.0f32; n_pixels]; num_c_load];

    let chunks_y = (height + chunk_y - 1) / chunk_y;
    let chunks_x = (width + chunk_x - 1) / chunk_x;
    let sep = &meta.dimension_separator;

    // Collect all chunk tasks across channels and `(cy, cx)` tiles, then decode them in parallel
    // across CPU worker threads using `std::thread::scope`.
    let mut chunk_jobs: Vec<(usize, usize, usize, PathBuf)> = Vec::new();
    for c_idx in 0..num_c_load {
        if let Some(only_c) = info.active_channel {
            if c_idx != only_c && num_c_load > 1 {
                continue;
            }
        }
        let c_chunk = c_idx / chunk_c;
        for cy_idx in 0..chunks_y {
            for cx_idx in 0..chunks_x {
                let rel_path = if meta.shape.len() == 2 {
                    format!("{cy_idx}{sep}{cx_idx}")
                } else {
                    format!("{c_chunk}{sep}{cy_idx}{sep}{cx_idx}")
                };
                let chunk_file = array_dir.join(rel_path);
                if chunk_file.is_file() {
                    chunk_jobs.push((c_idx, cy_idx, cx_idx, chunk_file));
                }
            }
        }
    }

    let comp_id = meta.compressor.as_ref().map(|c| c.id.as_str()).unwrap_or("");
    let dtype_str = meta.dtype.as_str();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let jobs_per_thread = (chunk_jobs.len() + num_threads - 1).max(1) / num_threads;

    let decoded_batches: Vec<IoResult<Vec<(usize, usize, usize, Vec<f32>)>>> =
        std::thread::scope(|s| {
            let handles: Vec<_> = chunk_jobs
                .chunks(jobs_per_thread.max(1))
                .map(|job_slice| {
                    s.spawn(move || {
                        let mut out_batch = Vec::with_capacity(job_slice.len());
                        for (c_idx, cy_idx, cx_idx, chunk_file) in job_slice {
                            let raw_file = fs::read(chunk_file)?;
                            let decompressed = match comp_id {
                                "blosc" => blosc_decompress(&raw_file)?,
                                "" => raw_file,
                                other => {
                                    return Err(IoError::OmeZarr(format!(
                                        "Unsupported Zarr compressor '{other}'"
                                    )))
                                }
                            };
                            let vals = decode_elements_to_f32(&decompressed, dtype_str)?;
                            out_batch.push((*c_idx, *cy_idx, *cx_idx, vals));
                        }
                        Ok(out_batch)
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(IoError::OmeZarr("Chunk worker panicked".into())))
                })
                .collect()
        });

    for batch_res in decoded_batches {
        for (c_idx, cy_idx, cx_idx, vals) in batch_res? {
            let y0 = cy_idx * chunk_y;
            let x0 = cx_idx * chunk_x;
            let valid_h = (height - y0).min(chunk_y);
            let valid_w = (width - x0).min(chunk_x);

            let plane = &mut channel_planes[c_idx];
            for ly in 0..valid_h {
                let src_row = ly * chunk_x;
                let dst_row = (y0 + ly) * width + x0;
                if src_row + valid_w <= vals.len() {
                    plane[dst_row..dst_row + valid_w]
                        .copy_from_slice(&vals[src_row..src_row + valid_w]);
                }
            }
        }
    }

    // For integer label masks (`<i4`), map non-zero label IDs via a golden-ratio hash so adjacent
    // cells/nuclei are visually distinct while preserving background = 0.0.
    let is_label_mask = info.active_label_group.is_some();
    for plane in &mut channel_planes {
        if is_label_mask {
            for v in plane.iter_mut() {
                let id = *v as u32;
                if id == 0 {
                    *v = 0.0;
                } else {
                    let h = ((id.wrapping_mul(2654435761)) & 0xFFFF) as f32 / 65535.0;
                    *v = 0.25 + 0.75 * h;
                }
            }
        } else {
            let max_v = plane.iter().copied().fold(0.0f32, f32::max).max(1.0);
            for v in plane.iter_mut() {
                *v = (*v / max_v).clamp(0.0, 1.0);
            }
        }
    }

    let mut luminance_f32 = vec![0.0f32; n_pixels];
    let mut rgba_f32 = vec![0.0f32; n_pixels * 4];

    match (num_c_load, info.active_channel) {
        (1, _) => {
            let p0 = &channel_planes[0];
            for i in 0..n_pixels {
                let v = p0[i];
                luminance_f32[i] = v;
                rgba_f32[i * 4] = v;
                rgba_f32[i * 4 + 1] = v;
                rgba_f32[i * 4 + 2] = v;
                rgba_f32[i * 4 + 3] = 1.0;
            }
        }
        (_, Some(ch_idx)) => {
            let p = &channel_planes[ch_idx.min(num_c_load - 1)];
            for i in 0..n_pixels {
                let v = p[i];
                luminance_f32[i] = v;
                rgba_f32[i * 4] = v;
                rgba_f32[i * 4 + 1] = v;
                rgba_f32[i * 4 + 2] = v;
                rgba_f32[i * 4 + 3] = 1.0;
            }
        }
        _ => {
            let p0 = &channel_planes[0];
            let p1 = &channel_planes[1];
            for i in 0..n_pixels {
                let c0 = p0[i];
                let c1 = p1[i];
                let lum = (0.65 * c0 + 0.65 * c1).max(c0.max(c1)).clamp(0.0, 1.0);
                luminance_f32[i] = lum;
                rgba_f32[i * 4] = c1;
                rgba_f32[i * 4 + 1] = c0;
                rgba_f32[i * 4 + 2] = c0;
                rgba_f32[i * 4 + 3] = 1.0;
            }
        }
    }

    let mut updated_info = info.clone();
    if updated_info.channel_labels.len() < channels {
        for c in updated_info.channel_labels.len()..channels {
            updated_info.channel_labels.push(format!("Ch {c}"));
        }
    }

    let mode_suffix = if let Some(ref lbl) = updated_info.active_label_group {
        format!("labels/{lbl}/{}", updated_info.active_level)
    } else if let Some(ch) = updated_info.active_channel {
        let ch_name = updated_info
            .channel_labels
            .get(ch)
            .cloned()
            .unwrap_or_else(|| format!("c{ch}"));
        format!("{} · {ch_name}", updated_info.active_level)
    } else {
        format!("{} · Composite", updated_info.active_level)
    };

    Ok(LoadedImage {
        name: format!("{} ({mode_suffix})", updated_info.container_name),
        path: Some(updated_info.root_path.clone()),
        width,
        height,
        channels,
        dtype_label: format!("ome-zarr {}", meta.dtype),
        luminance_f32,
        rgba_f32,
        ome_zarr: Some(updated_info),
    })
}

/// Scans `.data/` in the workspace for top-level `.ome.zarr` containers and standard image files.
pub fn discover_workspace_data_paths(workspace_dir: &Path) -> Vec<PathBuf> {
    let data_dir = workspace_dir.join(".data");
    let mut found = Vec::new();
    if !data_dir.is_dir() {
        return found;
    }

    if let Ok(entries) = fs::read_dir(&data_dir) {
        let mut top_paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        top_paths.sort();

        for p in top_paths {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.ends_with(".ome.zarr") && p.is_dir() {
                // Expose the single `.ome.zarr` container itself
                found.push(p);
            } else if p.is_file() {
                found.push(p);
            } else if p.is_dir() {
                if let Ok(sub_entries) = fs::read_dir(&p) {
                    let mut sub_paths: Vec<PathBuf> = sub_entries.flatten().map(|e| e.path()).collect();
                    sub_paths.sort();
                    for sp in sub_paths {
                        if sp.is_file() || sp.to_string_lossy().ends_with(".ome.zarr") {
                            found.push(sp);
                        }
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_workspace_ome_zarr_container_and_views() {
        let ws_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let zarr_path = ws_root.join(".data/TDTNI1_P4_M_SAL_S1_stack.ome.zarr");
        if !zarr_path.exists() {
            return;
        }

        let discovered = discover_workspace_data_paths(ws_root);
        assert_eq!(discovered, vec![zarr_path.clone()]);

        let img = load_ome_zarr(&zarr_path).expect("load default OME-Zarr container");
        assert_eq!(img.width, 1015);
        assert_eq!(img.height, 762);
        let max_lum = img.luminance_f32.iter().copied().fold(0.0f32, f32::max);
        assert!(max_lum > 0.5, "expected non-zero normalized luminance, got {max_lum}");

        let mut info = img.ome_zarr.expect("ome_zarr metadata present");
        assert_eq!(info.pyramid_levels, vec!["s0", "s1", "s2", "s3"]);
        assert_eq!(info.channel_labels, vec!["DAPI_deblur", "RFP"]);
        assert_eq!(info.label_groups, vec!["cells", "nuclei"]);
        assert_eq!(info.select_level_for_display_width(400.0), "s3");
        assert_eq!(info.select_level_for_display_width(900.0), "s2");
        assert_eq!(info.select_level_for_display_width(1800.0), "s1");
        assert_eq!(info.select_level_for_display_width(3200.0), "s0");

        // Test single-channel selection
        info.active_channel = Some(0);
        let ch0 = load_ome_zarr_with_options(&info).expect("load channel 0");
        assert_eq!(ch0.width, 1015);

        // Test label mask selection
        info.active_label_group = Some("nuclei".to_string());
        let lbl = load_ome_zarr_with_options(&info).expect("load nuclei labels");
        assert_eq!(lbl.width, 1015);
        assert!(lbl.luminance_f32.iter().any(|&v| v > 0.0));
    }
}
