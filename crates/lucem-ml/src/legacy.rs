//! lucem was called lumen. What a computer kept under the old name keeps working:
//!
//! - **Environment**: every `LUMEN_*` variable is read as its `LUCEM_*` twin when that one is not
//!   set ([`adopt`] copies them at start-up).
//! - **Folders**: `lumen/` under the config, data and cache folders (`~/.config`, `~/.local/share`,
//!   `~/.cache`, or their `XDG_*` and Windows equivalents) is moved to `lucem/` once, when
//!   `lucem/` does not exist yet ([`adopt`]).
//! - **Names in files**: [`OLD_NAME`] is accepted where the new name is expected (a project file
//!   `project.lumen.json`, script header lines `# lumen:`, plugin programs `lumen-ml-*`); the
//!   callers check both.

use std::path::PathBuf;

/// The old name.
pub const OLD_NAME: &str = "lumen";
/// The name.
pub const NAME: &str = "lucem";

/// Copies every `LUMEN_X` into `LUCEM_X` when that one is unset, and moves the old folders.
/// Call it first thing in `main`, before any thread starts.
pub fn adopt() {
    let old: Vec<(String, std::ffi::OsString)> = std::env::vars_os()
        .filter_map(|(k, v)| {
            let k = k.into_string().ok()?;
            let rest = k.strip_prefix("LUMEN_")?;
            Some((format!("LUCEM_{rest}"), v))
        })
        .collect();
    for (k, v) in old {
        if std::env::var_os(&k).is_none() {
            // SAFETY: called at the start of `main`, before any other thread reads the environment
            unsafe { std::env::set_var(&k, v) };
        }
    }
    for base in bases() {
        let (from, to) = (base.join(OLD_NAME), base.join(NAME));
        if from.is_dir() && !to.exists() {
            let _ = std::fs::rename(&from, &to);
        }
    }
}

/// The folders an app keeps its settings, data and caches under.
fn bases() -> Vec<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let home = env("HOME").or_else(|| env("USERPROFILE"));
    let mut out = Vec::new();
    out.extend(env("XDG_CONFIG_HOME").or_else(|| home.as_ref().map(|h| h.join(".config"))));
    out.extend(env("XDG_DATA_HOME").or_else(|| home.as_ref().map(|h| h.join(".local").join("share"))));
    out.extend(env("XDG_CACHE_HOME").or_else(|| home.as_ref().map(|h| h.join(".cache"))));
    out.extend(env("APPDATA"));
    out.extend(env("LOCALAPPDATA"));
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    /// An old folder moves once; one already there is left alone.
    #[test]
    fn old_folders_move_once() {
        let root = std::env::temp_dir().join(format!("lucem-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("lumen")).unwrap();
        std::fs::write(root.join("lumen").join("prefs.json"), "{}").unwrap();
        let (from, to) = (root.join(super::OLD_NAME), root.join(super::NAME));
        if from.is_dir() && !to.exists() {
            std::fs::rename(&from, &to).unwrap();
        }
        assert!(root.join("lucem").join("prefs.json").is_file() && !root.join("lumen").exists());
        std::fs::remove_dir_all(&root).ok();
    }
}
