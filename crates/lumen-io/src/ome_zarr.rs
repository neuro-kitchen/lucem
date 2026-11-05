//! Pure-Rust OME-Zarr v0.4 (`zarr_format: 2`) reader with Blosc-LZ4 + byte-unshuffle decoding.
//!
//! Supports opening either:
//! - An OME-Zarr root group (`.ome.zarr`), automatically selecting a responsive pyramid level (`s1`
//!   or `s0`), or
//! - A specific pyramid array directory (e.g., `.ome.zarr/s0`, `.ome.zarr/s1`, `.ome.zarr/s2`,
//!   `.ome.zarr/s3`, `.ome.zarr/labels/cells/s1`, `.ome.zarr/labels/nuclei/s1`).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{IoError, IoResult};
use crate::image_io::LoadedImage;

#[derive(Debug, Clone, Deserialize)]
struct ZarrCompressor {
    id: String,
    #[serde(default)]
    cname: Option<String>,
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
    let is_split = (flags & 0x10) != 0;
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

        let out_block_start = b_idx * bsize_default;
        let bsize = (nbytes - out_block_start).min(bsize_default);

        let num_streams = if is_split && typesize > 1 { typesize } else { 1 };
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

/// Resolves an `.ome.zarr` directory or pyramid sub-directory to the concrete directory containing
/// `.zarray`.
fn resolve_zarray_dir(path: &Path) -> IoResult<(PathBuf, String)> {
    let dir = if path.is_file() {
        path.parent().unwrap_or(path).to_path_buf()
    } else {
        path.to_path_buf()
    };

    if dir.join(".zarray").is_file() {
        let label = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("s0")
            .to_string();
        let parent_name = dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("ome.zarr");
        return Ok((dir.clone(), format!("{parent_name}/{label}")));
    }

    // If the user opened the root `.ome.zarr` directory, prefer `s1` (around 2031×1525) for
    // responsive interactive GPU exploration, falling back to `s0` or `s2`.
    for candidate in ["s1", "s0", "s2", "s3", "0"] {
        let sub = dir.join(candidate);
        if sub.join(".zarray").is_file() {
            let root_name = dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("ome.zarr");
            return Ok((sub, format!("{root_name}/{candidate}")));
        }
    }

    Err(IoError::OmeZarr(format!(
        "No .zarray found inside '{}'",
        path.display()
    )))
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

/// Returns `true` if `path` looks like an OME-Zarr directory or metadata file.
pub fn is_ome_zarr_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    s.contains(".ome.zarr")
        || path.join(".zarray").is_file()
        || path.join(".zgroup").is_file()
        || path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n == ".zarray" || n == ".zattrs" || n == ".zgroup")
}

/// Loads a 2D `[Y, X]` or 3D `[C, Y, X]` OME-Zarr array directory into a normalized [`LoadedImage`].
pub fn load_ome_zarr(path: &Path) -> IoResult<LoadedImage> {
    let (array_dir, display_name) = resolve_zarray_dir(path)?;
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
    let num_c_Load = channels.min(3);
    let mut channel_planes = vec![vec![0.0f32; n_pixels]; num_c_Load];

    let chunks_y = (height + chunk_y - 1) / chunk_y;
    let chunks_x = (width + chunk_x - 1) / chunk_x;
    let sep = &meta.dimension_separator;

    for c_idx in 0..num_c_Load {
        let c_chunk = c_idx / chunk_c;
        for cy_idx in 0..chunks_y {
            for cx_idx in 0..chunks_x {
                let rel_path = if meta.shape.len() == 2 {
                    format!("{cy_idx}{sep}{cx_idx}")
                } else {
                    format!("{c_chunk}{sep}{cy_idx}{sep}{cx_idx}")
                };
                let chunk_file = array_dir.join(rel_path);
                if !chunk_file.is_file() {
                    continue;
                }
                let raw_file = fs::read(&chunk_file)?;
                let decompressed = match meta.compressor.as_ref().map(|c| c.id.as_str()) {
                    Some("blosc") => blosc_decompress(&raw_file)?,
                    None => raw_file,
                    Some(other) => {
                        return Err(IoError::OmeZarr(format!("Unsupported Zarr compressor '{other}'")))
                    }
                };
                let vals = decode_elements_to_f32(&decompressed, &meta.dtype)?;

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
    }

    // Normalize each channel using its dynamic range (or OME 12-bit window 4095 for <u2 bio stacks)
    for plane in &mut channel_planes {
        let max_v = plane.iter().copied().fold(0.0f32, f32::max);
        let denom = if meta.dtype.ends_with("u2") && max_v > 255.0 && max_v <= 6000.0 {
            max_v.max(1.0)
        } else {
            max_v.max(1.0)
        };
        for v in plane.iter_mut() {
            *v = (*v / denom).clamp(0.0, 1.0);
        }
    }

    let mut luminance_f32 = vec![0.0f32; n_pixels];
    let mut rgba_f32 = vec![0.0f32; n_pixels * 4];

    if num_c_Load == 1 {
        let p0 = &channel_planes[0];
        for i in 0..n_pixels {
            let v = p0[i];
            luminance_f32[i] = v;
            rgba_f32[i * 4] = v;
            rgba_f32[i * 4 + 1] = v;
            rgba_f32[i * 4 + 2] = v;
            rgba_f32[i * 4 + 3] = 1.0;
        }
    } else {
        // 2-channel OME-Zarr (e.g., c0 = DAPI_deblur [Cyan], c1 = RFP [Red]) or 3-channel composite
        let p0 = &channel_planes[0];
        let p1 = &channel_planes[1];
        for i in 0..n_pixels {
            let c0 = p0[i];
            let c1 = p1[i];
            // Composite luminance preserves both DAPI nuclei and RFP soma/filaments
            let lum = (0.65 * c0 + 0.65 * c1).max(c0.max(c1)).clamp(0.0, 1.0);
            luminance_f32[i] = lum;
            rgba_f32[i * 4] = c1;
            rgba_f32[i * 4 + 1] = c0;
            rgba_f32[i * 4 + 2] = c0;
            rgba_f32[i * 4 + 3] = 1.0;
        }
    }

    Ok(LoadedImage {
        name: display_name,
        path: Some(array_dir),
        width,
        height,
        channels,
        dtype_label: format!("ome-zarr {}", meta.dtype),
        luminance_f32,
        rgba_f32,
    })
}

/// Scans `.data/` in the workspace for OME-Zarr pyramid datasets and standard image files.
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
                // Expose multiscale pyramid levels (`s2`, `s1`, `s0`) and label masks so the user can
                // benchmark different resolutions directly from the Startup Hub
                for sub in ["s2", "s1", "s0", "labels/nuclei/s2", "labels/cells/s2"] {
                    let candidate = p.join(sub);
                    if candidate.join(".zarray").is_file() {
                        found.push(candidate);
                    }
                }
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
