//! The app's samples as files: the volume and the phantoms, written as OME-Zarr with an
//! `index.json` naming them. A browser page opens these instead of generating them (seconds of CPU
//! there): the docs viewer from its site, a container's page from its server (`--samples`).

use std::path::Path;

use crate::error::{IoError, IoResult};
use crate::phantoms::{generate_phantom, PhantomKind};
use crate::pyramid::{sample_volume, Pyramid};
use crate::zarr_writer::{WriterOptions, ZarrWriter};

/// The sample volume's id (`samples/sample-volume.ome.zarr`).
pub const VOLUME_ID: &str = "sample-volume";

/// Writes every sample under `dir` (replacing older copies) and `index.json`; their file names.
pub fn write_all(dir: &Path) -> IoResult<Vec<String>> {
    std::fs::create_dir_all(dir)?;
    let mut written = Vec::new();
    write_one(dir, VOLUME_ID, &sample_volume(256, 128)?, &mut written)?;
    for kind in PhantomKind::ALL {
        let mut img = generate_phantom(*kind, 2048, 2048);
        img.name = kind.title().to_string();
        write_one(dir, &kind.id(), &Pyramid::from_loaded(img), &mut written)?;
    }
    std::fs::write(dir.join("index.json"), serde_json::to_vec_pretty(&written)?)?;
    Ok(written)
}

fn write_one(dir: &Path, id: &str, py: &Pyramid, written: &mut Vec<String>) -> IoResult<()> {
    let out = dir.join(format!("{id}.ome.zarr"));
    if out.exists() {
        std::fs::remove_dir_all(&out)?;
    }
    let (d, h, w) = (py.depth(), py.height(), py.width());
    let options = WriterOptions { name: py.name.clone(), channels: py.channels.iter().map(|c| c.label.clone()).collect(), shape: [d, h, w], chunks: [d.min(32), 256.min(h), 256.min(w)], pixel_size_um: py.pixel_size_um, z_spacing_um: py.z_spacing_um };
    let chunks = options.chunks;
    let writer = ZarrWriter::create::<f32>(&out, options)?;
    for b in lucem_core::tiling::schedule([d, h, w], chunks, [0; 3]) {
        for c in 0..py.channels.len() {
            let values = py.read_block(0, c, b.origin, b.size)?;
            writer.write(c, b.origin, b.size, &values)?;
        }
    }
    writer.finish::<f32>().map_err(|e| IoError::OmeZarr(format!("{id}: {e}")))?;
    compress(&out)?;
    written.push(format!("{id}.ome.zarr"));
    Ok(())
}

/// Compresses every chunk of the OME-Zarr at `dir` with zstd (in place; each `.zarray` says so):
/// a page downloads the samples, and the synthetic images shrink many times.
fn compress(dir: &Path) -> IoResult<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if path.is_dir() {
            compress(&path)?;
        } else if name == ".zarray" {
            let mut meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            meta["compressor"] = serde_json::json!({ "id": "zstd", "level": 1 });
            std::fs::write(&path, serde_json::to_vec_pretty(&meta)?)?;
        } else if !name.starts_with('.') && is_chunk_of_array(&path) {
            let raw = std::fs::read(&path)?;
            std::fs::write(&path, ruzstd::encoding::compress_to_vec(&raw[..], ruzstd::encoding::CompressionLevel::Fastest))?;
        }
    }
    Ok(())
}

/// A chunk file: somewhere under a folder holding a `.zarray` (an array, `0/`, `1/`, …).
fn is_chunk_of_array(path: &Path) -> bool {
    path.ancestors().skip(1).any(|a| a.join(".zarray").is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compressed_sample_reads_back_the_same() {
        let dir = std::env::temp_dir().join(format!("lucem-samples-zstd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut img = generate_phantom(PhantomKind::ALL[0], 300, 200);
        img.name = "t".into();
        let py = Pyramid::from_loaded(img);
        let mut written = Vec::new();
        write_one(&dir, "t", &py, &mut written).unwrap();
        let back = Pyramid::open(&dir.join("t.ome.zarr")).unwrap();
        let meta = std::fs::read_to_string(dir.join("t.ome.zarr/0/.zarray")).unwrap();
        assert!(meta.contains("zstd"), "{meta}");
        for c in 0..py.channels.len() {
            assert_eq!(back.read_block(0, c, [0, 0, 0], [1, 200, 300]).unwrap(), py.read_block(0, c, [0, 0, 0], [1, 200, 300]).unwrap());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
