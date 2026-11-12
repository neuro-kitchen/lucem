//! Finding the plugin programs: every executable named `lucem-ml-*` in, in order,
//! `$LUCEM_PLUGINS` (folders separated as in `PATH`), the user's plugin folder
//! (`$XDG_DATA_HOME/lucem/plugins` or `~/.local/share/lucem/plugins`), the folder of the running
//! program (an installed app with its plugins beside it, or `target/<profile>` in development)
//! and `PATH`. The first program of each name wins; each is asked for its manifest
//! (`<program> info`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::PluginInfo;

/// Plugin programs start with this.
pub const PROGRAM_PREFIX: &str = "lucem-ml-";
/// The prefix of plugin programs installed when lucem was called lumen.
pub const OLD_PROGRAM_PREFIX: &str = "lumen-ml-";

/// The folders searched, in order.
pub fn plugin_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(v) = std::env::var_os("LUCEM_PLUGINS") {
        dirs.extend(std::env::split_paths(&v));
    }
    let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")));
    if let Some(d) = data {
        dirs.push(d.join("lucem").join("plugins"));
    }
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir);
    }
    if let Some(p) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&p));
    }
    dirs
}

/// Whether `path` is a plugin program by its name.
fn is_plugin(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return false };
    let stem = name.strip_suffix(".exe").unwrap_or(name);
    // Build by-products (`lucem-ml-sam.d`) are not programs; `lumen-ml-*` are the same plugins
    // installed under the old name
    (stem.starts_with(PROGRAM_PREFIX) || stem.starts_with(OLD_PROGRAM_PREFIX)) && !stem.contains('.') && path.is_file()
}

/// The plugin programs found, by name.
pub fn programs() -> BTreeMap<String, PathBuf> {
    let mut out = BTreeMap::new();
    for dir in plugin_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if is_plugin(&p) {
                // An old-named program answers to the new name (a new-named one comes first)
                let name = p.file_name().map(|n| n.to_string_lossy().replacen(OLD_PROGRAM_PREFIX, PROGRAM_PREFIX, 1)).unwrap_or_default();
                let old = p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(OLD_PROGRAM_PREFIX));
                match out.get(&name) {
                    Some(_) if old => {}
                    _ => {
                        out.insert(name, p);
                    }
                }
            }
        }
    }
    out
}

/// Every plugin found and its manifest; a program that does not answer is reported as an error
/// (its name and why), not left out silently.
pub fn discover() -> Vec<Result<PluginInfo, String>> {
    programs()
        .into_iter()
        .map(|(name, path)| {
            let out = Command::new(&path).arg("info").output().map_err(|e| format!("{name}: {e}"))?;
            if !out.status.success() {
                return Err(format!("{name}: {}", String::from_utf8_lossy(&out.stderr).trim()));
            }
            let mut info: PluginInfo = serde_json::from_slice(&out.stdout).map_err(|e| format!("{name}: its manifest does not read: {e}"))?;
            info.program = path.to_string_lossy().into_owned();
            Ok(info)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_lucem_ml_programs_are_plugins() {
        let dir = std::env::temp_dir().join(format!("lucem-ml-discover-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["lucem-ml-sam", "lucem-ml-sam.d", "lucem-app", "notes.txt"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let found: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).filter(|p| is_plugin(p)).map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(found, vec!["lucem-ml-sam".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
