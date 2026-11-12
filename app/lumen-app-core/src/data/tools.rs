//! The programs lumen calls but does not ship: Python (and uv) for Python scripts, cargo and rustc
//! for Rust scripts, ruff to format and check Python. Each is found, in order, at the path set in
//! Settings › Tools, at its environment variable (`LUMEN_PYTHON`…), or on the `PATH`. Nothing is
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

    /// What lumen uses it for (Settings).
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
            Tool::Python => "LUMEN_PYTHON",
            Tool::Uv => "LUMEN_UV",
            Tool::Ruff => "LUMEN_RUFF",
            Tool::Cargo => "LUMEN_CARGO",
            Tool::Rustc => "LUMEN_RUSTC",
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
pub fn python_command() -> Command {
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
        let dir = std::env::temp_dir().join(format!("lumen-tools-{}", std::process::id()));
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
