//! The programs lucem calls but does not ship: Python (and uv) for Python scripts, cargo and rustc
//! for Rust scripts, ruff to format and check Python. Each is found, in order, at the path set in
//! Settings › Tools, at its environment variable (`LUCEM_PYTHON`…), or on the `PATH`. Nothing is
//! downloaded.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

/// The paths set in Settings (`None`: found as described in the module).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolPaths {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uv: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ruff: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cargo: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rustc: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    Python,
    Uv,
    Ruff,
    Cargo,
    Rustc,
}

impl Tool {
    pub const ALL: [Tool; 5] = [Tool::Python, Tool::Uv, Tool::Ruff, Tool::Cargo, Tool::Rustc];

    pub fn title(self) -> &'static str {
        match self {
            Tool::Python => "Python",
            Tool::Uv => "uv",
            Tool::Ruff => "Ruff",
            Tool::Cargo => "Cargo",
            Tool::Rustc => "rustc",
        }
    }

    /// What lucem uses it for (Settings).
    pub fn what(self) -> &'static str {
        match self {
            Tool::Python => "Runs Python script nodes and table code. Needs NumPy for script nodes, unless uv is set.",
            Tool::Uv => "Optional. When set, Python scripts run through uv, which provides NumPy and each script's own packages.",
            Tool::Ruff => "Optional. Formats and checks Python scripts in the code view.",
            Tool::Cargo => "Builds Rust scripts that use crates from crates.io.",
            Tool::Rustc => "Compiles Rust scripts.",
        }
    }

    /// Its program name on the `PATH`.
    fn program(self) -> &'static str {
        match self {
            Tool::Python => "python3",
            Tool::Uv => "uv",
            Tool::Ruff => "ruff",
            Tool::Cargo => "cargo",
            Tool::Rustc => "rustc",
        }
    }

    pub fn env(self) -> &'static str {
        match self {
            Tool::Python => "LUCEM_PYTHON",
            Tool::Uv => "LUCEM_UV",
            Tool::Ruff => "LUCEM_RUFF",
            Tool::Cargo => "LUCEM_CARGO",
            Tool::Rustc => "LUCEM_RUSTC",
        }
    }

    fn set_path(self, p: &ToolPaths) -> Option<&PathBuf> {
        match self {
            Tool::Python => p.python.as_ref(),
            Tool::Uv => p.uv.as_ref(),
            Tool::Ruff => p.ruff.as_ref(),
            Tool::Cargo => p.cargo.as_ref(),
            Tool::Rustc => p.rustc.as_ref(),
        }
    }

    /// The path field of `p` for this tool (Settings writes it).
    pub fn slot(self, p: &mut ToolPaths) -> &mut Option<PathBuf> {
        match self {
            Tool::Python => &mut p.python,
            Tool::Uv => &mut p.uv,
            Tool::Ruff => &mut p.ruff,
            Tool::Cargo => &mut p.cargo,
            Tool::Rustc => &mut p.rustc,
        }
    }
}

static PATHS: RwLock<ToolPaths> = RwLock::new(ToolPaths { python: None, uv: None, ruff: None, cargo: None, rustc: None });

/// Uses the paths set in Settings (the store calls it when the preferences load or change).
pub fn set(paths: &ToolPaths) {
    *PATHS.write().unwrap_or_else(|e| e.into_inner()) = paths.clone();
}

/// Where a tool comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum Found {
    /// Set in Settings.
    Settings(PathBuf),
    /// Named by its environment variable.
    Env(PathBuf),
    /// On the `PATH`.
    Path(PathBuf),
    Missing,
}

impl Found {
    pub fn path(&self) -> Option<&Path> {
        match self {
            Found::Settings(p) | Found::Env(p) | Found::Path(p) => Some(p),
            Found::Missing => None,
        }
    }
}

/// Finds `tool`: Settings, then its environment variable, then the `PATH`.
pub fn find(tool: Tool) -> Found {
    if let Some(p) = tool.set_path(&PATHS.read().unwrap_or_else(|e| e.into_inner())) {
        return Found::Settings(p.clone());
    }
    if let Some(p) = std::env::var_os(tool.env()).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(p);
        return Found::Env(on_path(&p).unwrap_or(p));
    }
    on_path(Path::new(tool.program())).map_or(Found::Missing, Found::Path)
}

/// The program to run for `tool` (its name, to fail with a clear message, when it is missing).
pub fn program(tool: Tool) -> PathBuf {
    find(tool).path().map_or_else(|| PathBuf::from(tool.program()), Path::to_path_buf)
}

/// The command that runs a Python file: uv (with NumPy, and the chosen Python if one is set) when
/// uv is found, else Python itself.
/// The project's scripts folder (where its environments live), set when a project opens.
static SCRIPTS: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Sets the project's scripts folder (`None`: no project; the user's scripts folder).
pub fn set_scripts_dir(dir: Option<PathBuf>) {
    *SCRIPTS.write().unwrap_or_else(|e| e.into_inner()) = dir;
}

/// The scripts folder of the project open, else the user's.
pub fn scripts_dir() -> PathBuf {
    SCRIPTS.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| crate::data::script::scripts_dir(None))
}

/// lucem's own Python environment, for projects that have none (`<config>/python/.venv`).
pub fn default_env_dir() -> PathBuf {
    crate::data::prefs::Prefs::dir().unwrap_or_else(std::env::temp_dir).join("python")
}

/// The Python environment scripts run in: the project's (`scripts/.venv`), else the one active
/// (`$VIRTUAL_ENV`), else lucem's default; `None` when none was made.
pub fn python_env() -> Option<kitchen_shell::env::PythonEnv> {
    kitchen_shell::env::PythonEnv::find(Some(&scripts_dir()), &default_env_dir().join(".venv"))
}

/// Makes or syncs the project's Python environment with uv (`scripts/pyproject.toml`,
/// `uv.lock`, `.venv`; NumPy, which script nodes need, in a new one).
pub fn sync_project_env() -> Result<String, String> {
    let dir = scripts_dir();
    let name = dir.parent().and_then(|p| p.file_name()).map_or("lucem-scripts".to_string(), |n| format!("{}-scripts", n.to_string_lossy()));
    kitchen_shell::env::sync_python(&dir, &name, &["numpy"])
}

/// Makes or syncs lucem's default Python environment (for projects without their own).
pub fn sync_default_env() -> Result<String, String> {
    kitchen_shell::env::sync_python(&default_env_dir(), "lucem-default", &["numpy"])
}

/// The Rust toolchain the project pins (`scripts/rust-toolchain.toml`).
pub fn rust_toolchain() -> Option<String> {
    kitchen_shell::env::rust_toolchain(&scripts_dir())
}

/// Pins the compiler found now in `scripts/rust-toolchain.toml` (rustup then builds the
/// project's Rust scripts with it, on any computer).
pub fn pin_rust() -> Result<String, String> {
    let dir = scripts_dir();
    let v = kitchen_shell::env::rustc_version().ok_or("rustc was not found")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("rust-toolchain.toml"), format!("# The compiler this project's Rust scripts are built with (rustup reads it).\n[toolchain]\nchannel = \"{v}\"\n")).map_err(|e| e.to_string())?;
    Ok(v)
}

/// Where Rust builds run, so rustup uses the project's pin: the scripts folder when it pins one.
pub fn rust_dir() -> Option<PathBuf> {
    let d = scripts_dir();
    d.join("rust-toolchain.toml").is_file().then_some(d)
}

/// How Python runs: the path set in Settings when there is one; else the project's environment
/// (or the active or default one); else through uv with NumPy; else `python3` on the `PATH`.
pub fn python_command() -> Command {
    if Tool::Python.set_path(&PATHS.read().unwrap_or_else(|e| e.into_inner())).is_none()
        && let Some(env) = python_env()
    {
        return env.command();
    }
    match find(Tool::Uv) {
        Found::Missing => Command::new(program(Tool::Python)),
        uv => {
            let mut c = Command::new(uv.path().expect("found"));
            c.args(["run", "--no-project", "--quiet", "--with", "numpy"]);
            if let Some(py) = Tool::Python.set_path(&PATHS.read().unwrap_or_else(|e| e.into_inner())) {
                c.arg("--python").arg(py);
            }
            c.arg("python");
            c
        }
    }
}

/// `tool --version`'s first line (run it off the UI thread).
pub fn version(tool: Tool) -> Result<String, String> {
    let program = program(tool);
    let out = Command::new(&program).arg("--version").output().map_err(|_| "Not found".to_string())?;
    let text = if out.stdout.is_empty() { out.stderr } else { out.stdout };
    let line = String::from_utf8_lossy(&text).lines().next().unwrap_or_default().trim().to_string();
    if out.status.success() && !line.is_empty() { Ok(line) } else { Err(format!("{} did not run", tool.title())) }
}

/// A program name or path as the `PATH` resolves it (a path with a folder in it as it is).
fn on_path(program: &Path) -> Option<PathBuf> {
    if program.components().count() > 1 {
        return program.is_file().then(|| program.to_path_buf());
    }
    let exts: Vec<String> = if cfg!(windows) { vec![String::new(), ".exe".into()] } else { vec![String::new()] };
    std::env::split_paths(&std::env::var_os("PATH")?).flat_map(|dir| exts.iter().map(move |e| dir.join(format!("{}{e}", program.display())))).find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_set_in_settings_comes_first() {
        let dir = std::env::temp_dir().join(format!("lucem-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ruff = dir.join("ruff");
        std::fs::write(&ruff, "").unwrap();
        set(&ToolPaths { ruff: Some(ruff.clone()), ..ToolPaths::default() });
        assert_eq!(find(Tool::Ruff), Found::Settings(ruff));
        set(&ToolPaths::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tool_paths_read_back() {
        let p = ToolPaths { python: Some("/opt/py/bin/python3".into()), ..ToolPaths::default() };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, r#"{"python":"/opt/py/bin/python3"}"#);
        assert_eq!(serde_json::from_str::<ToolPaths>(&json).unwrap(), p);
        assert_eq!(serde_json::from_str::<ToolPaths>("{}").unwrap(), ToolPaths::default());
    }
}
