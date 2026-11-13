//! `lucem-hub`: verified downloads of published artifacts (model weights) and their local cache,
//! as dsp-kitchen's `dspk-hub`. All network access of lucem's models is here.
//!
//! A file is fetched to a temporary name, hashed while it streams, and committed to the cache only
//! when its SHA-256 and size match; a file without a hash is refused. The cache is
//! `$LUCEM_MODELS`, else `$XDG_CACHE_HOME/lucem/models`, else `~/.cache/lucem/models`, one folder
//! per model family.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The cache folder of a model family.
pub fn cache_dir(family: &str) -> PathBuf {
    let base = std::env::var_os("LUCEM_MODELS").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(std::env::temp_dir)
            .join("lucem")
            .join("models")
    });
    base.join(family)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-256 of a file (lowercase hex).
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Whether `path` holds the expected file (size first, then hash).
pub fn is_valid(path: &Path, sha256: &str, bytes: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|m| bytes == 0 || m.len() == bytes) && sha256_file(path).is_ok_and(|h| h.eq_ignore_ascii_case(sha256))
}

/// Reads a file only if its hash matches (what a model loads).
pub fn read_verified(path: &Path, sha256: &str) -> Result<Vec<u8>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = hex(&Sha256::digest(&bytes));
    if got.eq_ignore_ascii_case(sha256) {
        Ok(bytes)
    } else {
        Err(format!("{}: the file is not the published one (SHA-256 {got}, expected {sha256}); delete it and install again", path.display()))
    }
}

/// Downloads `url` to `dest` unless a valid copy is there; `progress(done, total)` follows the
/// bytes. The file is committed only when its SHA-256 (and size, when given) match.
pub fn fetch(url: &str, dest: &Path, sha256: &str, bytes: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String> {
    if sha256.trim().is_empty() {
        return Err(format!("{url}: no hash published; refused"));
    }
    if is_valid(dest, sha256, bytes) {
        return Ok(dest.to_path_buf());
    }
    if let Some(d) = dest.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let part = dest.with_extension("part");
    let resp = ureq::get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let total = resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).unwrap_or(bytes);
    let mut reader = resp.into_body().into_reader();
    let mut out = File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    let mut h = Sha256::new();
    let (mut done, mut buf) = (0u64, vec![0u8; 1 << 20]);
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{url}: {e}"))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        done += n as u64;
        progress(done, total);
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    let got = hex(&h.finalize());
    if !got.eq_ignore_ascii_case(sha256) || (bytes != 0 && done != bytes) {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{url}: the download does not match (SHA-256 {got}, {done} bytes; expected {sha256}, {bytes} bytes)"));
    }
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    Ok(dest.to_path_buf())
}

/// Downloads a `.tgz` (verified like [`fetch`]) and unpacks it into `dest_dir`, unless
/// `dest_dir` already holds what it unpacked (a `.complete` mark is written last).
pub fn fetch_archive(url: &str, archive: &Path, sha256: &str, bytes: u64, dest_dir: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String> {
    let mark = dest_dir.join(".complete");
    if mark.is_file() {
        return Ok(dest_dir.to_path_buf());
    }
    fetch(url, archive, sha256, bytes, progress)?;
    let file = File::open(archive).map_err(|e| format!("{}: {e}", archive.display()))?;
    let _ = std::fs::remove_dir_all(dest_dir);
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("{}: {e}", dest_dir.display()))?;
    tar::Archive::new(flate2::read::GzDecoder::new(file)).unpack(dest_dir).map_err(|e| format!("{}: {e}", archive.display()))?;
    std::fs::write(&mark, sha256).map_err(|e| e.to_string())?;
    // The archive is not needed once unpacked
    let _ = std::fs::remove_file(archive);
    Ok(dest_dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_read_only_with_its_hash() {
        let p = std::env::temp_dir().join(format!("lucem-hub-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let h = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_file(&p).unwrap(), h);
        assert!(read_verified(&p, h).is_ok() && read_verified(&p, &"0".repeat(64)).is_err());
        assert!(is_valid(&p, h, 3) && !is_valid(&p, h, 4));
        assert!(fetch("https://example.invalid/x", &p, "", 0, &mut |_, _| {}).is_err());
        let _ = std::fs::remove_file(&p);
    }
}
