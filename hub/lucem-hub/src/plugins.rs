//! Verified installation of `lucem-ml-*` plugin binaries from GitHub Releases, mirroring
//! `dsp-kitchen`'s `dspk-hub::sorters` architecture.
//!
//! Model plugins (`lucem-ml-stardist`, `lucem-ml-instanseg`, `lucem-ml-sam`) are never linked
//! into `lucem` and are not distributed as PyPI wheels: each release attaches prebuilt binaries
//! and a SHA-256 manifest (`lucem-ml-plugins-manifest.json`), and [`install`] downloads,
//! verifies, and installs the binary for this platform into the user plugin directory
//! (`$XDG_DATA_HOME/lucem/plugins` or `~/.local/share/lucem/plugins`, which `lucem_ml::discover`
//! searches automatically).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fetch;

/// Known ML model plugin families shipped as release binaries.
pub const KNOWN_PLUGINS: &[&str] = &["stardist", "instanseg", "sam"];

/// Default release download base URL: `<base>/v<version>/<file>`.
pub const RELEASE_BASE_URL: &str = "https://github.com/neuro-kitchen/lucem/releases/download";

/// Environment variable overriding [`RELEASE_BASE_URL`] (mirrors or `file://` for offline tests).
pub const RELEASE_URL_ENV: &str = "LUCEM_PLUGINS_RELEASE_URL";

/// Manifest file attached to each GitHub Release.
pub const MANIFEST_FILE: &str = "lucem-ml-plugins-manifest.json";

/// One platform's plugin binary entry in [`PluginsManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginAsset {
    pub file: String,
    pub sha256: String,
    pub size: u64,
}

/// `lucem-ml-plugins-manifest.json`: keyed by plugin program (`lucem-ml-stardist`, ...) then
/// Rust target triple (`x86_64-unknown-linux-gnu`, ...).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PluginsManifest {
    pub version: String,
    pub plugins: BTreeMap<String, BTreeMap<String, PluginAsset>>,
}

/// Target triple for this host if prebuilt release binaries are published for it.
pub fn host_target() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        _ => None,
    }
}

/// Normalizes `"stardist"` or `"lucem-ml-stardist"` to `"lucem-ml-stardist"`.
pub fn program_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.starts_with("lucem-ml-") {
        trimmed.to_string()
    } else {
        format!("lucem-ml-{trimmed}")
    }
}

/// Executable file name on this platform (e.g. `lucem-ml-stardist` or `lucem-ml-stardist.exe`).
pub fn program_file_name(name: &str) -> String {
    format!("{}{}", program_name(name), std::env::consts::EXE_SUFFIX)
}

/// User plugin install directory (`$XDG_DATA_HOME/lucem/plugins`, else
/// `~/.local/share/lucem/plugins`, or `%LOCALAPPDATA%\lucem\plugins` on Windows).
pub fn install_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .or_else(|| std::env::var_os("LOCALAPPDATA"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("lucem").join("plugins")
}

/// Release base URL for `version` (`<base>/v<version>`).
pub fn release_url(version: &str) -> String {
    let base = std::env::var(RELEASE_URL_ENV)
        .ok()
        .filter(|b| !b.trim().is_empty())
        .unwrap_or_else(|| RELEASE_BASE_URL.to_string());
    format!("{}/v{version}", base.trim().trim_end_matches('/'))
}

/// Downloads and parses [`MANIFEST_FILE`] for `version`.
pub fn fetch_manifest(version: &str) -> Result<PluginsManifest, String> {
    let url = format!("{}/{MANIFEST_FILE}", release_url(version));
    let text = match url.strip_prefix("file://") {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("no plugin manifest at {url}: {e}"))?,
        None => ureq::get(&url)
            .call()
            .map_err(|e| format!("no plugin manifest at {url} (is v{version} released with ML plugin binaries?): {e}"))?
            .into_body()
            .read_to_string()
            .map_err(|e| format!("failed to read {url}: {e}"))?,
    };
    let manifest: PluginsManifest = serde_json::from_str(&text).map_err(|e| format!("{url} is not a plugin manifest: {e}"))?;
    if manifest.version != version {
        return Err(format!("{url} describes version {}, not {version}", manifest.version));
    }
    Ok(manifest)
}

/// Downloads, verifies (SHA-256 + size), and installs `plugin` (`"stardist"`, `"instanseg"`,
/// `"sam"`, or `"lucem-ml-*"`) for `version` into [`install_dir`].
pub fn install(plugin: &str, version: &str, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String> {
    let Some(target) = host_target() else {
        return Err(format!(
            "no prebuilt {} binary is released for {} {}; build from source (`cargo build --release -p {}`)",
            program_name(plugin),
            std::env::consts::OS,
            std::env::consts::ARCH,
            program_name(plugin),
        ));
    };
    let prog = program_name(plugin);
    let manifest = fetch_manifest(version)?;
    let Some(targets) = manifest.plugins.get(&prog) else {
        let avail: Vec<&str> = manifest.plugins.keys().map(String::as_str).collect();
        return Err(format!("release v{version} does not include {prog} (available: {})", avail.join(", ")));
    };
    let Some(asset) = targets.get(target) else {
        return Err(format!("release v{version} has no {prog} build for {target}"));
    };
    let dest = install_dir().join(program_file_name(&prog));
    let url = format!("{}/{}", release_url(version), asset.file);
    install_from_url(&url, &dest, &asset.sha256, asset.size, progress)
}

fn install_from_url(url: &str, dest: &Path, sha256: &str, size: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf, String> {
    if let Some(local) = url.strip_prefix("file://") {
        let bytes = crate::read_verified(Path::new(local), sha256)?;
        if size != 0 && bytes.len() as u64 != size {
            return Err(format!("{url}: expected {size} bytes, got {}", bytes.len()));
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(dest, &bytes).map_err(|e| format!("{}: {e}", dest.display()))?;
        progress(bytes.len() as u64, bytes.len() as u64);
    } else {
        fetch(url, dest, sha256, size, progress)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dest, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("failed to make {} executable: {e}", dest.display()))?;
    }
    Ok(dest.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips_and_normalizes_names() {
        assert_eq!(program_name("stardist"), "lucem-ml-stardist");
        assert_eq!(program_name("lucem-ml-sam"), "lucem-ml-sam");
        let mut manifest = PluginsManifest { version: "0.0.6".into(), ..Default::default() };
        let mut targets = BTreeMap::new();
        targets.insert(
            "x86_64-unknown-linux-gnu".into(),
            PluginAsset { file: "lucem-ml-stardist-x86_64-unknown-linux-gnu".into(), sha256: "ab".repeat(32), size: 42 },
        );
        manifest.plugins.insert("lucem-ml-stardist".into(), targets);
        let text = serde_json::to_string(&manifest).unwrap();
        assert_eq!(serde_json::from_str::<PluginsManifest>(&text).unwrap(), manifest);
    }
}
