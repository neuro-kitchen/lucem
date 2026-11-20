//! User code in Python or Rust, run as a separate program (so a script never takes the app down):
//!
//! - **Script nodes** of the Pipeline (Blender's Script node): a file whose header declares its
//!   inputs, its output and its numbers —
//!   ```text
//!   # lucem: input image: image        (Rust: // lucem: …)
//!   # lucem: input mask: mask
//!   # lucem: param sigma = 2.0
//!   # lucem: output mask
//!   ```
//!   Python: `def run(image, mask, sigma)` takes NumPy arrays (`float32` images, `uint32`
//!   masks and labels) and returns one. Rust: `pub fn run(inputs: &Inputs, params: &Params) ->
//!   Result<Array, String>` (types in the generated harness, [`RUST_PRELUDE`]). Arrays travel as
//!   `.npy` files in a scratch folder; a result is cached by the file and its inputs.
//! - **Table code** of Analyze (VS Code's Data Wrangler): a cell run on the shown table, `df`;
//!   Python gets a Polars frame (else pandas, else a list of rows), Rust a small [`Table`
//!   API](RUST_TABLE_PRELUDE). Tables travel as CSV.
//!
//! The programs come from [`crate::data::tools`]: Settings › Tools, else `$LUCEM_PYTHON`… , else
//! the `PATH`. Python is `python3` and needs NumPy for script nodes (or uv, which provides it).
//! Rust is `rustc`; a Rust file is compiled once per version of it, in the cache folder. A Rust
//! script may use crates from crates.io, one header line each —
//!   `// lucem: crate rayon = "1"` or `// lucem: crate image = { version = "0.25", features = ["png"] }`
//! — and is then built with `cargo`; the crates compile once, shared by every
//! script.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use lucem_pipeline::PortType;

use crate::domain::graph::{ScriptParam, ScriptPort};

/// The languages user code is written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Python,
    Rust,
}

impl Language {
    pub const ALL: [Language; 2] = [Language::Python, Language::Rust];

    pub fn of(path: &Path) -> Option<Language> {
        match path.extension()?.to_str()? {
            "py" => Some(Language::Python),
            "rs" => Some(Language::Rust),
            _ => None,
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Language::Python => "Python",
            Language::Rust => "Rust",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Language::Python => "py",
            Language::Rust => "rs",
        }
    }

    fn comment(self) -> &'static str {
        match self {
            Language::Python => "#",
            Language::Rust => "//",
        }
    }
}

// ----------------------------------------------------------------------------
// The header
// ----------------------------------------------------------------------------

/// What a script declares via `lucem:` header comments.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub name: Option<String>,
    pub family: Option<String>,
    pub description: Option<String>,
    pub halo: usize,
    pub inputs: Vec<ScriptPort>,
    pub output: PortType,
    pub params: Vec<ScriptParam>,
}

fn port_of(name: &str) -> Option<PortType> {
    match name.trim().to_lowercase().as_str() {
        "image" => Some(PortType::Image),
        "mask" => Some(PortType::Mask),
        "labels" => Some(PortType::Labels),
        _ => None,
    }
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    s.strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')))
        .unwrap_or(s)
        .to_string()
}

/// Reads the `lucem:` lines of a script (comments starting with `#` or `//`).
pub fn read_header(text: &str) -> Result<Header, String> {
    let mut h = Header {
        name: None,
        family: None,
        description: None,
        halo: 0,
        inputs: Vec::new(),
        output: PortType::Image,
        params: Vec::new(),
    };
    for (i, line) in text.lines().enumerate() {
        let l = line.trim_start();
        let l = l.strip_prefix("//").or_else(|| l.strip_prefix('#')).unwrap_or("").trim();
        // `lumen:` is the old name's header
        let Some(rest) = l.strip_prefix("lucem:").or_else(|| l.strip_prefix("lumen:")) else { continue };
        let rest = rest.trim();
        let bad = |what: &str| format!("line {}: {what}: {line:?}", i + 1);
        if let Some(r) = rest.strip_prefix("name")
            && let Some((_, v)) = r.split_once('=')
        {
            let v = unquote(v);
            if !v.is_empty() {
                h.name = Some(v);
            }
        } else if let Some(r) = rest.strip_prefix("family")
            && let Some((_, v)) = r.split_once('=')
        {
            let v = unquote(v);
            if !v.is_empty() {
                h.family = Some(v);
            }
        } else if let Some(r) = rest.strip_prefix("description")
            && let Some((_, v)) = r.split_once('=')
        {
            let v = unquote(v);
            if !v.is_empty() {
                h.description = Some(v);
            }
        } else if let Some(r) = rest.strip_prefix("halo")
            && let Some((_, v)) = r.split_once('=')
        {
            h.halo = v.trim().parse::<usize>().map_err(|_| bad("halo must be a non-negative integer"))?;
        } else if let Some(r) = rest.strip_prefix("input ") {
            let (name, port) = r.split_once(':').ok_or_else(|| bad("an input is `input NAME: image|mask|labels`"))?;
            let port = port_of(port).ok_or_else(|| bad("the type is image, mask or labels"))?;
            h.inputs.push(ScriptPort { name: name.trim().to_string(), port });
        } else if let Some(r) = rest.strip_prefix("output") {
            let t = r.rsplit(':').next().unwrap_or(r);
            h.output = port_of(t).ok_or_else(|| bad("the output is image, mask or labels"))?;
        } else if let Some(r) = rest.strip_prefix("param ") {
            let (name, value) = r.split_once('=').ok_or_else(|| bad("a number is `param NAME = VALUE`"))?;
            // Allow optional trailing range/unit hint like `param sigma = 2.0 (0.2..20.0)`
            let num_token = value.trim().split_whitespace().next().unwrap_or("");
            let value = num_token.parse::<f64>().map_err(|_| bad("the value is a number"))?;
            h.params.push(ScriptParam { name: name.trim().to_string(), value });
        }
    }
    Ok(h)
}

/// Extracts Python package dependencies declared via PEP 723 inline script metadata
/// (`# /// script` ... `dependencies = ["scipy", "scikit-image"]`) or `# lucem: pip <pkg>`.
pub fn read_python_deps(text: &str) -> Vec<String> {
    let mut deps = Vec::new();
    let mut in_pep723 = false;
    let mut in_deps = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(after) = trimmed.strip_prefix('#').map(str::trim) {
            if after == "/// script" {
                in_pep723 = true;
                continue;
            }
            if after == "///" && in_pep723 {
                in_pep723 = false;
                in_deps = false;
                continue;
            }
            if let Some(pkg) = after.strip_prefix("lucem:").map(str::trim).and_then(|r| r.strip_prefix("pip ")) {
                let pkg = unquote(pkg);
                if !pkg.is_empty() {
                    deps.push(pkg);
                }
                continue;
            }
            if in_pep723 {
                if after.starts_with("dependencies") && after.contains('[') {
                    in_deps = true;
                }
                if in_deps {
                    // Collect quoted strings inside `[...]`
                    let mut rest = after;
                    while let Some(start) = rest.find('"') {
                        let tail = &rest[start + 1..];
                        let Some(end) = tail.find('"') else { break };
                        let pkg = tail[..end].trim();
                        if !pkg.is_empty() && pkg != "dependencies" {
                            deps.push(pkg.to_string());
                        }
                        rest = &tail[end + 1..];
                    }
                    if after.contains(']') {
                        in_deps = false;
                    }
                }
            }
        }
    }
    deps.sort();
    deps.dedup();
    deps
}

/// A drop-in community or user script op discovered in a plugin `ops/` folder.
#[derive(Debug, Clone)]
pub struct DiscoveredScript {
    pub path: PathBuf,
    pub name: String,
    pub family: String,
    pub description: String,
    pub header: Header,
}

/// Folders scanned for drop-in script ops (`$LUCEM_PLUGINS/ops` and `<config>/lucem/plugins/ops`).
pub fn plugin_ops_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(env_dirs) = std::env::var("LUCEM_PLUGINS") {
        for p in std::env::split_paths(&env_dirs) {
            dirs.push(p.join("ops"));
        }
    }
    if let Some(cfg) = crate::data::prefs::Prefs::dir() {
        dirs.push(cfg.join("plugins").join("ops"));
    }
    dirs
}

/// Scans [`plugin_ops_dirs`] for `.py` and `.rs` files with valid `lucem:` headers.
pub fn discover_plugin_ops() -> Vec<DiscoveredScript> {
    let mut found = Vec::new();
    for dir in plugin_ops_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if Language::of(&path).is_none() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(header) = read_header(&text) else { continue };
            let fallback = path.file_stem().map_or_else(|| "Script".to_string(), |s| s.to_string_lossy().into_owned());
            let name = header.name.clone().unwrap_or(fallback);
            let family = header.family.clone().unwrap_or_else(|| "Plugin".to_string());
            let description = header.description.clone().unwrap_or_default();
            found.push(DiscoveredScript { path, name, family, description, header });
        }
    }
    found.sort_by(|a, b| a.family.cmp(&b.family).then_with(|| a.name.cmp(&b.name)));
    found
}

/// A new script's text: the header and one example step.
pub fn template(lang: Language) -> String {
    match lang {
        Language::Python => r#"# A lucem script node. The header declares its inputs, numbers and output; lucem
# reads it again when the file is saved, and the node runs again.
# lucem: input image: image
# lucem: param threshold = 0.5
# lucem: output mask
#
# `image` is a 2-D float32 NumPy array (values 0-1); return a 2-D array of the output type.
import numpy as np


def run(image, threshold):
    return (image > threshold).astype(np.uint32)
"#
        .to_string(),
        Language::Rust => r#"// A lucem script node in Rust. The header declares its inputs, numbers and output;
// lucem compiles the file again when it is saved (the standard library and the `Array` type of
// the harness; crates from crates.io with header lines such as `// lucem: crate rayon = "1"`).
// lucem: input image: image
// lucem: param threshold = 0.5
// lucem: output mask
//
// `inputs["image"]` is an `Array` (width, height, data: Vec<f32>, row by row); masks are 0 / 1.

pub fn run(inputs: &Inputs, params: &Params) -> Result<Array, String> {
    let image = inputs.get("image").ok_or("no image")?;
    let t = params.get("threshold").copied().unwrap_or(0.5) as f32;
    Ok(image.map(|v| if v > t { 1.0 } else { 0.0 }))
}
"#
        .to_string(),
    }
}

/// Where new scripts are written: the project's `scripts/` folder, else the user's.
pub fn scripts_dir(project: Option<&Path>) -> PathBuf {
    match project {
        Some(p) => p.join("scripts"),
        None => crate::data::prefs::Prefs::dir().unwrap_or_else(std::env::temp_dir).join("scripts"),
    }
}

/// Writes a new script from the template, named `name_<n>.<ext>` (never over an existing file).
pub fn new_script(dir: &Path, lang: Language) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let path = (1..10_000).map(|n| dir.join(format!("script_{n}.{}", lang.extension()))).find(|p| !p.exists()).ok_or("no free name")?;
    std::fs::write(&path, template(lang)).map_err(|e| e.to_string())?;
    Ok(path)
}

/// Opens a file in the system's editor.
pub fn open_in_editor(path: &Path) -> Result<(), String> {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    Command::new(program).arg(path).spawn().map(|_| ()).map_err(|e| format!("{program}: {e}"))
}

// ----------------------------------------------------------------------------
// Arrays as .npy
// ----------------------------------------------------------------------------

/// A 2-D raster on the host, row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct HostArray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

fn npy_bytes(a: &HostArray, integer: bool) -> Vec<u8> {
    let descr = if integer { "<u4" } else { "<f4" };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({}, {}), }}", a.height, a.width);
    // Magic (6) + version (2) + length (2) + header + newline: a multiple of 64
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + a.data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &v in &a.data {
        if integer {
            out.extend_from_slice(&(v.max(0.0).round() as u32).to_le_bytes());
        } else {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Reads a 2-D `.npy` of float32, float64, uint8, uint16, uint32, int32, int64 or bool.
pub fn read_npy(bytes: &[u8]) -> Result<HostArray, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let (hlen, start) = if bytes[6] == 1 { (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10) } else { (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12) };
    let header = std::str::from_utf8(bytes.get(start..start + hlen).ok_or("short .npy header")?).map_err(|e| e.to_string())?;
    let field = |key: &str| header.split(&format!("'{key}':")).nth(1).map(str::trim);
    let descr = field("descr").and_then(|d| d.split('\'').nth(1)).ok_or("no dtype")?.to_string();
    if field("fortran_order").is_some_and(|f| f.starts_with("True")) {
        return Err("Fortran-ordered arrays are not supported (use np.ascontiguousarray)".into());
    }
    let shape = field("shape").and_then(|s| s.split(')').next()).ok_or("no shape")?;
    let dims: Vec<usize> = shape.trim_start_matches('(').split(',').filter_map(|d| d.trim().parse().ok()).collect();
    let (h, w) = match dims.as_slice() {
        [h, w] => (*h, *w),
        [n] => (1, *n),
        _ => return Err(format!("the result must be 2-D, it is {dims:?}")),
    };
    let body = &bytes[start + hlen..];
    let n = w * h;
    let size = |s: usize| if body.len() >= n * s { Ok(()) } else { Err("the .npy file is short".to_string()) };
    let data: Vec<f32> = match descr.trim_start_matches(['<', '|', '=']) {
        "f4" => {
            size(4)?;
            body.as_chunks::<4>().0.iter().take(n).map(|c| f32::from_le_bytes(*c)).collect()
        }
        "f8" => {
            size(8)?;
            body.as_chunks::<8>().0.iter().take(n).map(|c| f64::from_le_bytes(*c) as f32).collect()
        }
        "u1" | "b1" => {
            size(1)?;
            body.iter().take(n).map(|&v| v as f32).collect()
        }
        "u2" => {
            size(2)?;
            body.as_chunks::<2>().0.iter().take(n).map(|c| u16::from_le_bytes(*c) as f32).collect()
        }
        "u4" => {
            size(4)?;
            body.as_chunks::<4>().0.iter().take(n).map(|c| u32::from_le_bytes(*c) as f32).collect()
        }
        "i4" => {
            size(4)?;
            body.as_chunks::<4>().0.iter().take(n).map(|c| i32::from_le_bytes(*c) as f32).collect()
        }
        "i8" => {
            size(8)?;
            body.as_chunks::<8>().0.iter().take(n).map(|c| i64::from_le_bytes(*c) as f32).collect()
        }
        other => return Err(format!("dtype {other} is not supported (use float32 or uint32)")),
    };
    Ok(HostArray { width: w, height: h, data })
}

// ----------------------------------------------------------------------------
// Running
// ----------------------------------------------------------------------------

fn rustc() -> PathBuf {
    crate::data::tools::program(crate::data::tools::Tool::Rustc)
}

/// The cache folder of compiled Rust scripts and results.
fn cache_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache"))).unwrap_or_else(std::env::temp_dir);
    base.join("lucem").join("scripts")
}

/// A fresh scratch folder.
fn scratch() -> Result<PathBuf, String> {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = crate::data::python_worker::shm_scratch_root().join(format!("lucem-run-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn run_program(cmd: &mut Command, what: &str) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("{what} could not start: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        // The end of a traceback says what went wrong
        let tail: Vec<&str> = err.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
        Err(format!("{what} failed:\n{}", tail.join("\n")))
    }
}

fn cargo() -> PathBuf {
    crate::data::tools::program(crate::data::tools::Tool::Cargo)
}

/// The crates a Rust script asks for: `NAME = SPEC` of each `lucem: crate` line, as a Cargo.toml
/// dependency line.
pub fn read_crates(text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let l = line.trim_start();
        let l = l.strip_prefix("//").or_else(|| l.strip_prefix('#')).unwrap_or("").trim();
        let Some(r) = l.strip_prefix("lucem:").map(str::trim).and_then(|r| r.strip_prefix("crate ")) else { continue };
        let (name, spec) = r.split_once('=').ok_or_else(|| format!("line {}: a crate is `crate NAME = \"VERSION\"`: {line:?}", i + 1))?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(format!("line {}: {name:?} is not a crate name", i + 1));
        }
        out.push(format!("{name} = {}", spec.trim()));
    }
    Ok(out)
}

/// Compiles `source` once (by its content and crates), returns the program: with rustc, or as a
/// Cargo project when it uses crates (one build folder for all scripts, so a crate compiles once).
fn compile_rust(source: &str, crates: &[String]) -> Result<PathBuf, String> {
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    crates.hash(&mut h);
    let id = format!("{:016x}", h.finish());
    let dir = cache_dir().join(&id);
    let bin = dir.join(if cfg!(windows) { "script.exe" } else { "script" });
    if bin.exists() {
        return Ok(bin);
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    if crates.is_empty() {
        let src = dir.join("main.rs");
        std::fs::write(&src, source).map_err(|e| e.to_string())?;
        let mut c = Command::new(rustc());
        if let Some(d) = crate::data::tools::rust_dir() {
            c.current_dir(d);
        }
        run_program(c.args(["--edition", "2021", "-O", "--crate-name", "lucem_script", "-o"]).arg(&bin).arg(&src), "rustc")?;
        return Ok(bin);
    }
    let package = format!("lucem_script_{id}");
    std::fs::create_dir_all(dir.join("src")).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("src").join("main.rs"), source).map_err(|e| e.to_string())?;
    let manifest = format!("[package]\nname = \"{package}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n[dependencies]\n{}\n\n[workspace]\n", crates.join("\n"));
    std::fs::write(dir.join("Cargo.toml"), manifest).map_err(|e| e.to_string())?;
    let target = cache_dir().join("target");
    let mut c = Command::new(cargo());
    // In the project's scripts folder, so rustup uses its toolchain pin
    if let Some(d) = crate::data::tools::rust_dir() {
        c.current_dir(d);
    }
    run_program(c.args(["build", "--release", "--quiet", "--manifest-path"]).arg(dir.join("Cargo.toml")).env("CARGO_TARGET_DIR", &target), "cargo")?;
    let built = target.join("release").join(if cfg!(windows) { format!("{package}.exe") } else { package });
    std::fs::rename(&built, &bin).or_else(|_| std::fs::copy(&built, &bin).map(|_| ())).map_err(|e| format!("{}: {e}", crate::domain::paths::short_path(&built)))?;
    Ok(bin)
}

/// What the Rust harness of script nodes gives the user's file.
pub const RUST_PRELUDE: &str = r#"#![allow(dead_code, unused)]
use std::collections::HashMap;

/// A 2-D raster, row by row.
#[derive(Clone, Debug)]
pub struct Array { pub width: usize, pub height: usize, pub data: Vec<f32> }

impl Array {
    pub fn new(width: usize, height: usize) -> Self { Self { width, height, data: vec![0.0; width * height] } }
    pub fn get(&self, x: usize, y: usize) -> f32 { self.data[y * self.width + x] }
    pub fn set(&mut self, x: usize, y: usize, v: f32) { self.data[y * self.width + x] = v; }
    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self { Self { width: self.width, height: self.height, data: self.data.iter().map(|&v| f(v)).collect() } }
}

pub type Inputs = HashMap<String, Array>;
pub type Params = HashMap<String, f64>;

mod npy {
    use super::Array;
    pub fn read(b: &[u8]) -> Array {
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let head = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        let shape = head.split("'shape':").nth(1).unwrap().split(')').next().unwrap();
        let d: Vec<usize> = shape.trim().trim_start_matches('(').split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let int = head.contains("<u4");
        let data = b[10 + hl..].chunks_exact(4).map(|c| { let a = [c[0], c[1], c[2], c[3]]; if int { u32::from_le_bytes(a) as f32 } else { f32::from_le_bytes(a) } }).collect();
        Array { width: d[1], height: d[0], data }
    }
    pub fn write(a: &Array) -> Vec<u8> {
        let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", a.height, a.width);
        while (10 + h.len() + 1) % 64 != 0 { h.push(' '); }
        h.push('\n');
        let mut o = b"\x93NUMPY\x01\x00".to_vec();
        o.extend_from_slice(&(h.len() as u16).to_le_bytes());
        o.extend_from_slice(h.as_bytes());
        for v in &a.data { o.extend_from_slice(&v.to_le_bytes()); }
        o
    }
}

mod user {
    use super::*;
    include!("__USER_FILE__");
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("folder"));
    let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap();
    let (mut inputs, mut params) = (Inputs::new(), Params::new());
    for l in meta.lines() {
        let p: Vec<&str> = l.split_whitespace().collect();
        match p.as_slice() {
            ["input", n] => { inputs.insert(n.to_string(), npy::read(&std::fs::read(dir.join(format!("{n}.npy"))).unwrap())); }
            ["param", n, v] => { params.insert(n.to_string(), v.parse().unwrap_or(0.0)); }
            _ => {}
        }
    }
    match user::run(&inputs, &params) {
        Ok(a) => std::fs::write(dir.join("output.npy"), npy::write(&a)).unwrap(),
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}
"#;

const PYTHON_NODE_RUNNER: &str = r#"import importlib.util, sys
import numpy as np
script, folder = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("lucem_script", script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
args, dtype = {}, "float32"
for line in open(folder + "/meta.txt"):
    p = line.split()
    if p[:1] == ["input"]:
        args[p[1]] = np.load(folder + "/" + p[1] + ".npy")
    elif p[:1] == ["param"]:
        args[p[1]] = float(p[2])
    elif p[:1] == ["dtype"]:
        dtype = p[1]
out = mod.run(**args)
if isinstance(out, dict):
    out = next(iter(out.values()))
np.save(folder + "/output.npy", np.ascontiguousarray(np.asarray(out).astype(dtype)))
"#;

pub use crate::data::python_worker::{
    ensure_managed_venv, managed_venvs_dir, restart_python_workers, run_in_warm_worker, set_warm_workers_enabled, warm_worker_count, warm_workers_enabled,
};

/// Results of script nodes by (file, its version, inputs, numbers): a camera move does not run
/// the script again.
fn results_cache() -> &'static Mutex<HashMap<u64, HostArray>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, HostArray>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Runs the script node at `path` on `inputs` (by name) with `params`, giving its output.
pub fn run_node(path: &Path, inputs: &[(String, PortType, HostArray)], params: &[ScriptParam], output: PortType) -> Result<HostArray, String> {
    let lang = Language::of(path).ok_or_else(|| format!("{}: a script is a .py or .rs file", crate::domain::paths::short_path(&path)))?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", crate::domain::paths::short_path(&path)))?;
    let mut h = DefaultHasher::new();
    (path, &text).hash(&mut h);
    for (n, _, a) in inputs {
        n.hash(&mut h);
        (a.width, a.height).hash(&mut h);
        for v in &a.data {
            v.to_bits().hash(&mut h);
        }
    }
    for p in params {
        (&p.name, p.value.to_bits()).hash(&mut h);
    }
    let key = h.finish();
    if let Some(hit) = crate::platform::lock(results_cache()).get(&key).cloned() {
        return Ok(hit);
    }
    let dir = scratch()?;
    let mut meta = String::new();
    for (name, port, a) in inputs {
        std::fs::write(dir.join(format!("{name}.npy")), npy_bytes(a, *port != PortType::Image)).map_err(|e| e.to_string())?;
        meta.push_str(&format!("input {name}\n"));
    }
    for p in params {
        meta.push_str(&format!("param {} {}\n", p.name, p.value));
    }
    meta.push_str(&format!("dtype {}\n", if output == PortType::Image { "float32" } else { "uint32" }));
    std::fs::write(dir.join("meta.txt"), meta).map_err(|e| e.to_string())?;
    let name = path.file_name().map_or_else(|| "script".into(), |n| n.to_string_lossy().into_owned());
    let result = (|| {
        match lang {
            Language::Python => {
                let extra_deps = read_python_deps(&text);
                let has_uv = Command::new(crate::data::tools::program(crate::data::tools::Tool::Uv))
                    .arg("--version")
                    .output()
                    .is_ok_and(|o| o.status.success());
                let (env_key, make_cmd): (String, Box<dyn Fn() -> Command>) = if !extra_deps.is_empty() && has_uv {
                    let py = ensure_managed_venv(&extra_deps)?;
                    let key = format!("venv:{}", py.display());
                    (
                        key,
                        Box::new(move || {
                            let mut c = Command::new(&py);
                            c.arg("-I");
                            c
                        }),
                    )
                } else {
                    ("default".to_string(), Box::new(crate::data::tools::python_command))
                };
                if warm_workers_enabled() {
                    run_in_warm_worker(&env_key, make_cmd.as_ref(), path, &dir)?;
                } else {
                    let runner = dir.join("runner.py");
                    std::fs::write(&runner, PYTHON_NODE_RUNNER).map_err(|e| e.to_string())?;
                    run_program(make_cmd().arg(&runner).arg(path).arg(&dir), &name)?;
                }
            }
            Language::Rust => {
                let abs = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
                let source = RUST_PRELUDE.replace("__USER_FILE__", &abs.to_string_lossy().replace('\\', "\\\\")) + &format!("\n// {:x}\n", key_of(&text));
                let bin = compile_rust(&source, &read_crates(&text)?)?;
                run_program(Command::new(bin).arg(&dir), &name)?;
            }
        }
        read_npy(&std::fs::read(dir.join("output.npy")).map_err(|e| format!("{name} wrote no result: {e}"))?)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    let out = result?;
    {
        let mut c = crate::platform::lock(results_cache());
        if c.len() > 64 {
            c.clear();
        }
        c.insert(key, out.clone());
    }
    Ok(out)
}

fn key_of(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

// ----------------------------------------------------------------------------
// Code on tables
// ----------------------------------------------------------------------------

const PYTHON_TABLE_RUNNER: &str = r#"import sys
folder = sys.argv[1]
code = open(folder + "/cell.py").read()
ns = {}
try:
    import polars as pl
    df = pl.read_csv(folder + "/in.csv")
    ns["pl"], kind = pl, "polars"
except ImportError:
    try:
        import pandas as pd
        df = pd.read_csv(folder + "/in.csv")
        ns["pd"], kind = pd, "pandas"
    except ImportError:
        import csv
        df = list(csv.DictReader(open(folder + "/in.csv")))
        kind = "rows"
ns["df"] = df
exec(compile(code, "cell", "exec"), ns)
out = ns.get("result", ns["df"])
if hasattr(out, "write_csv"):
    out.write_csv(folder + "/out.csv")
elif hasattr(out, "to_csv"):
    out.to_csv(folder + "/out.csv", index=False)
else:
    import csv
    rows = list(out)
    with open(folder + "/out.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
"#;

/// What a Rust table cell is given: `df`, a [`Table`] (columns of numbers or text).
pub const RUST_TABLE_PRELUDE: &str = r#"#![allow(dead_code, unused, unused_mut)]
/// A column: numbers (`NaN` where empty) or text.
#[derive(Clone, Debug)]
pub enum Col { Num(Vec<f64>), Text(Vec<String>) }

/// A table: named columns of one length.
#[derive(Clone, Debug, Default)]
pub struct Table { pub names: Vec<String>, pub cols: Vec<Col> }

impl Table {
    pub fn len(&self) -> usize { self.cols.first().map_or(0, |c| match c { Col::Num(v) => v.len(), Col::Text(v) => v.len() }) }
    fn at(&self, name: &str) -> usize { self.names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no column {name:?}; columns: {:?}", self.names)) }
    /// A column of numbers.
    pub fn num(&self, name: &str) -> Vec<f64> { match &self.cols[self.at(name)] { Col::Num(v) => v.clone(), Col::Text(v) => v.iter().map(|s| s.parse().unwrap_or(f64::NAN)).collect() } }
    /// A column as text.
    pub fn text(&self, name: &str) -> Vec<String> { match &self.cols[self.at(name)] { Col::Text(v) => v.clone(), Col::Num(v) => v.iter().map(|x| x.to_string()).collect() } }
    /// The rows where `keep` is true.
    pub fn filter(&self, keep: &[bool]) -> Table {
        let pick = |i: usize| keep.get(i).copied().unwrap_or(false);
        Table { names: self.names.clone(), cols: self.cols.iter().map(|c| match c {
            Col::Num(v) => Col::Num(v.iter().enumerate().filter(|(i, _)| pick(*i)).map(|(_, x)| *x).collect()),
            Col::Text(v) => Col::Text(v.iter().enumerate().filter(|(i, _)| pick(*i)).map(|(_, x)| x.clone()).collect()),
        }).collect() }
    }
    /// Adds (or replaces) a column of numbers.
    pub fn with_num(mut self, name: &str, v: Vec<f64>) -> Table { self.put(name, Col::Num(v)); self }
    /// Adds (or replaces) a column of text.
    pub fn with_text(mut self, name: &str, v: Vec<String>) -> Table { self.put(name, Col::Text(v)); self }
    fn put(&mut self, name: &str, c: Col) { match self.names.iter().position(|n| n == name) { Some(i) => self.cols[i] = c, None => { self.names.push(name.into()); self.cols.push(c); } } }
    /// Only these columns, in this order.
    pub fn select(&self, names: &[&str]) -> Table { Table { names: names.iter().map(|s| s.to_string()).collect(), cols: names.iter().map(|n| self.cols[self.at(n)].clone()).collect() } }
    /// Sorted by a column.
    pub fn sort_by(&self, name: &str, descending: bool) -> Table {
        let c = &self.cols[self.at(name)];
        let mut idx: Vec<usize> = (0..self.len()).collect();
        match c { Col::Num(v) => idx.sort_by(|&a, &b| v[a].total_cmp(&v[b])), Col::Text(v) => idx.sort_by(|&a, &b| v[a].cmp(&v[b])) }
        if descending { idx.reverse(); }
        Table { names: self.names.clone(), cols: self.cols.iter().map(|c| match c { Col::Num(v) => Col::Num(idx.iter().map(|&i| v[i]).collect()), Col::Text(v) => Col::Text(idx.iter().map(|&i| v[i].clone()).collect()) }).collect() }
    }
}

fn parse(text: &str) -> Table {
    let mut lines = text.lines();
    let names: Vec<String> = lines.next().unwrap_or("").split(',').map(|s| s.trim_matches('"').to_string()).collect();
    let rows: Vec<Vec<String>> = lines.filter(|l| !l.is_empty()).map(|l| l.split(',').map(|s| s.trim_matches('"').to_string()).collect()).collect();
    let cols = (0..names.len()).map(|c| {
        let v: Vec<String> = rows.iter().map(|r| r.get(c).cloned().unwrap_or_default()).collect();
        if v.iter().all(|s| s.is_empty() || s == "null" || s.parse::<f64>().is_ok()) { Col::Num(v.iter().map(|s| s.parse().unwrap_or(f64::NAN)).collect()) } else { Col::Text(v) }
    }).collect();
    Table { names, cols }
}

fn write(t: &Table) -> String {
    let mut o = t.names.join(",") + "\n";
    for i in 0..t.len() {
        let row: Vec<String> = t.cols.iter().map(|c| match c { Col::Num(v) => if v[i].is_nan() { String::new() } else { v[i].to_string() }, Col::Text(v) => if v[i].contains(',') { format!("\"{}\"", v[i]) } else { v[i].clone() } }).collect();
        o += &(row.join(",") + "\n");
    }
    o
}

fn run(mut df: Table) -> Result<Table, String> {
__USER_CODE__
    #[allow(unreachable_code)]
    Ok(df)
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("folder"));
    let t = parse(&std::fs::read_to_string(dir.join("in.csv")).unwrap());
    match run(t) {
        Ok(out) => std::fs::write(dir.join("out.csv"), write(&out)).unwrap(),
        Err(e) => { eprintln!("{e}"); std::process::exit(1); }
    }
}
"#;

/// Starter code for a table cell.
pub fn table_template(lang: Language) -> &'static str {
    match lang {
        Language::Python => "# `df` is the table (Polars, else pandas). Leave the result in `df` (or `result`).\ndf = df.filter(pl.col(\"class\") == \"Region\") if \"pl\" in dir() else df\n",
        Language::Rust => "// `df` is a Table: df.num(\"col\"), df.text(\"col\"), df.filter(&keep), df.with_num(name, v), df.sort_by(name, desc).\nlet keep: Vec<bool> = df.text(\"class\").iter().map(|c| c == \"Region\").collect();\ndf = df.filter(&keep);\n",
    }
}

/// Runs a table cell on `csv` (with a header row), giving the result as CSV.
pub fn run_table(lang: Language, code: &str, csv: &str) -> Result<String, String> {
    let dir = scratch()?;
    std::fs::write(dir.join("in.csv"), csv).map_err(|e| e.to_string())?;
    let result = (|| {
        match lang {
            Language::Python => {
                std::fs::write(dir.join("cell.py"), code).map_err(|e| e.to_string())?;
                let runner = dir.join("runner.py");
                std::fs::write(&runner, PYTHON_TABLE_RUNNER).map_err(|e| e.to_string())?;
                run_program(crate::data::tools::python_command().arg(&runner).arg(&dir), "Python")?;
            }
            Language::Rust => {
                let bin = compile_rust(&RUST_TABLE_PRELUDE.replace("__USER_CODE__", code), &read_crates(code)?)?;
                run_program(Command::new(bin).arg(&dir), "Rust")?;
            }
        }
        std::fs::read_to_string(dir.join("out.csv")).map_err(|e| format!("no result table: {e}"))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// A whole script of table steps, for export (each step as written, in order).
pub fn export_steps(lang: Language, steps: &[String]) -> String {
    let c = lang.comment();
    let mut out = format!("{c} Table steps exported from lucem's Analyze ({}).\n", lang.title());
    for (i, s) in steps.iter().enumerate() {
        out.push_str(&format!("\n{c} Step {}\n{}\n", i + 1, s.trim_end()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_declares_ports_numbers_and_output() {
        let h = read_header(&template(Language::Python)).unwrap();
        assert_eq!(h.inputs, vec![ScriptPort { name: "image".into(), port: PortType::Image }]);
        assert_eq!(h.output, PortType::Mask);
        assert_eq!(h.params, vec![ScriptParam { name: "threshold".into(), value: 0.5 }]);
        let r = read_header(&template(Language::Rust)).unwrap();
        assert_eq!(r, h);
        assert!(read_header("# lucem: input x: cube").is_err());
        assert_eq!(read_crates("// lucem: crate rayon = \"1\"\n// lucem: crate image = { version = \"0.25\", features = [\"png\"] }\nfn main() {}").unwrap(), vec!["rayon = \"1\"".to_string(), "image = { version = \"0.25\", features = [\"png\"] }".to_string()]);
        assert!(read_crates("// lucem: crate bad name = \"1\"").is_err());
        assert!(read_header("// lucem: crate rayon = \"1\"\n// lucem: output mask").is_ok());
    }

    fn has(cmd: impl AsRef<std::ffi::OsStr>, args: &[&str]) -> bool {
        Command::new(cmd).args(args).output().is_ok_and(|o| o.status.success())
    }

    /// Whether Python (through uv when it is set) runs with `args`.
    fn has_python(args: &[&str]) -> bool {
        crate::data::tools::python_command().args(args).output().is_ok_and(|o| o.status.success())
    }

    fn run_template(lang: Language) -> HostArray {
        let dir = std::env::temp_dir().join(format!("lucem-test-scripts-{}-{}", std::process::id(), lang.extension()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = new_script(&dir, lang).unwrap();
        let h = read_header(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let image = HostArray { width: 3, height: 2, data: vec![0.1, 0.9, 0.4, 0.6, 0.2, 1.0] };
        let out = run_node(&path, &[("image".into(), PortType::Image, image)], &h.params, h.output).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    /// The Rust template, compiled and run (skipped without rustc).
    #[test]
    fn the_rust_template_runs() {
        if !has(rustc(), &["--version"]) {
            eprintln!("skipped: no rustc");
            return;
        }
        let out = run_template(Language::Rust);
        assert_eq!((out.width, out.height), (3, 2));
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    /// A Rust script with a crate from crates.io, built with cargo (needs the network: run with
    /// `--ignored`).
    #[test]
    #[ignore = "downloads a crate"]
    fn a_rust_script_uses_a_crate() {
        let dir = std::env::temp_dir().join(format!("lucem-test-crate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("crate_node.rs");
        std::fs::write(&path, "// lucem: input image: image\n// lucem: crate libm = \"0.2\"\n// lucem: output image\npub fn run(inputs: &Inputs, _: &Params) -> Result<Array, String> {\n    let image = inputs.get(\"image\").ok_or(\"no image\")?;\n    Ok(image.map(|v| libm::floorf(v * 10.0)))\n}\n").unwrap();
        let h = read_header(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let image = HostArray { width: 3, height: 1, data: vec![0.14, 0.95, 0.5] };
        let out = run_node(&path, &[("image".into(), PortType::Image, image)], &h.params, h.output).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(out.data, vec![1.0, 9.0, 5.0]);
    }

    /// The Python template (skipped without python3 and NumPy).
    #[test]
    fn the_python_template_runs() {
        if !has_python(&["-c", "import numpy"]) {
            eprintln!("skipped: no python3 with NumPy");
            return;
        }
        let out = run_template(Language::Python);
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    /// Table code in Rust and Python (each skipped when its tool is missing).
    #[test]
    fn table_cells_run() {
        let csv = "id,class,area\na1,Region,12.5\na2,Tumor,3\n";
        if has(rustc(), &["--version"]) {
            let out = run_table(Language::Rust, table_template(Language::Rust), csv).unwrap();
            assert!(out.contains("a1") && !out.contains("a2"), "{out}");
        }
        if has_python(&["--version"]) {
            let out = run_table(Language::Python, "result = df\n", csv).unwrap();
            assert!(out.contains("a2"), "{out}");
        }
    }

    #[test]
    fn npy_round_trips() {
        let a = HostArray { width: 3, height: 2, data: vec![0.0, 1.5, 2.0, 3.0, 4.0, 5.25] };
        assert_eq!(read_npy(&npy_bytes(&a, false)).unwrap(), a);
        let m = HostArray { width: 2, height: 2, data: vec![0.0, 1.0, 1.0, 0.0] };
        assert_eq!(read_npy(&npy_bytes(&m, true)).unwrap(), m);
    }

    #[test]
    fn plugin_header_and_pep723_deps_parse() {
        let src = r#"# /// script
# dependencies = ["numpy", "scipy", "scikit-image"]
# ///
# lucem: name = "Membrane Enhance"
# lucem: family = "Filter"
# lucem: description = "DoG membrane enhancement"
# lucem: halo = 12
# lucem: input image: image
# lucem: param sigma = 1.5
# lucem: output image
"#;
        let h = read_header(src).unwrap();
        assert_eq!(h.name.as_deref(), Some("Membrane Enhance"));
        assert_eq!(h.family.as_deref(), Some("Filter"));
        assert_eq!(h.description.as_deref(), Some("DoG membrane enhancement"));
        assert_eq!(h.halo, 12);
        assert_eq!(h.inputs.len(), 1);
        assert_eq!(h.params.len(), 1);
        assert_eq!(read_python_deps(src), vec!["numpy", "scikit-image", "scipy"]);
    }
}

/// The lines of the script `file` (its file name) that an error names, with the message: Python
/// tracebacks (`File ".../name.py", line 12` then the error on the last line) and Rust compiler
/// messages (`--> .../name.rs:12:5` under `error[...]: message`). Other lines are left out.
pub fn error_lines(file: &str, text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let last = lines.iter().rev().find(|l| !l.trim().is_empty()).map(|l| l.trim().to_string()).unwrap_or_default();
    let mut out: Vec<(usize, String)> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let t = l.trim();
        // Python: File "/x/name.py", line 12, in run
        if let Some(rest) = t.strip_prefix("File \"")
            && let Some((path, tail)) = rest.split_once('"')
            && path.ends_with(file)
            && let Some(n) = tail.split("line ").nth(1).and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next()).and_then(|s| s.parse::<usize>().ok())
        {
            out.push((n, last.clone()));
        }
        // Rust: --> /x/name.rs:12:5, under "error[E0308]: mismatched types"
        if let Some(rest) = t.strip_prefix("--> ")
            && let Some((path, pos)) = rest.split_once(':')
            && (path.ends_with(file) || path.ends_with("main.rs"))
            && let Some(n) = pos.split(':').next().and_then(|s| s.parse::<usize>().ok())
        {
            let msg = lines[..i].iter().rev().find(|p| p.trim_start().starts_with("error")).map(|p| p.trim().to_string()).unwrap_or_else(|| last.clone());
            out.push((n, msg));
        }
    }
    out.dedup();
    out
}

#[cfg(test)]
mod error_line_tests {
    #[test]
    fn python_and_rust_errors_name_their_lines() {
        let py = "Traceback (most recent call last):\n  File \"/tmp/run.py\", line 40, in <module>\n  File \"/p/scripts/blur.py\", line 7, in run\n    return x / 0\nZeroDivisionError: division by zero\n";
        assert_eq!(super::error_lines("blur.py", py), vec![(7, "ZeroDivisionError: division by zero".to_string())]);
        let rs = "error[E0308]: mismatched types\n --> /c/lucem/abc/main.rs:12:5\n   |\n";
        assert_eq!(super::error_lines("blur.rs", rs), vec![(12, "error[E0308]: mismatched types".to_string())]);
    }
}
