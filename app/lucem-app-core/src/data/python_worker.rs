//! Persistent warm Python worker pool and managed `uv` `.venv` cache for Lucem script ops.
//!
//! Keeping a lightweight stdlib Python worker alive per `(venv, deps)` environment avoids paying
//! interpreter startup and heavy ML framework imports (`torch`, `scipy`, `numpy`) on every tile
//! or pan/zoom, while still isolating user code in a separate OS process.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{BufRead as _, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Returns a RAM-backed shared-memory directory (`/dev/shm/lucem-scripts` on Linux when writable,
/// else the OS temp dir) so Rust ↔ Python tile buffers transfer via zero-copy `mmap` in RAM
/// without touching disk.
pub fn shm_scratch_root() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        let shm = Path::new("/dev/shm");
        if shm.is_dir() {
            let root = shm.join("lucem-scripts");
            if std::fs::create_dir_all(&root).is_ok() {
                return root;
            }
        }
    }
    std::env::temp_dir()
}

/// Long-lived stdlib Python worker speaking JSON-lines on stdin/stdout. Keeps Python, NumPy,
/// and user ML frameworks (PyTorch/ONNX/SciPy) warm between tile runs, maps shared-memory inputs
/// zero-copy via `mmap_mode="r"`, calls optional `init()` once per script version, and reloads
/// the module automatically when the script changes on disk.
const PYTHON_WARM_WORKER: &str = r#"import hashlib, importlib.util, json, sys, traceback
import numpy as np

loaded = {}  # script_path -> (sha, module)

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue
    try:
        req = json.loads(raw_line)
        script = req["script"]
        folder = req["folder"]
        with open(script, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        entry = loaded.get(script)
        if entry is None or entry[0] != sha:
            spec = importlib.util.spec_from_file_location("lucem_user_" + sha[:12], script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "init") and callable(mod.init):
                mod.init()
            loaded[script] = (sha, mod)
        else:
            mod = entry[1]
        args, dtype = {}, "float32"
        with open(folder + "/meta.txt") as mf:
            for line in mf:
                p = line.split()
                if p[:1] == ["input"]:
                    # Zero-copy memory-mapped view directly into /dev/shm RAM buffer
                    args[p[1]] = np.load(folder + "/" + p[1] + ".npy", mmap_mode="r")
                elif p[:1] == ["param"]:
                    args[p[1]] = float(p[2])
                elif p[:1] == ["dtype"]:
                    dtype = p[1]
        out = mod.run(**args)
        if isinstance(out, dict):
            out = next(iter(out.values()))
        np.save(folder + "/output.npy", np.ascontiguousarray(np.asarray(out).astype(dtype)))
        sys.stdout.write(json.dumps({"ok": True}) + "\n")
        sys.stdout.flush()
    except Exception:
        tb = traceback.format_exc()
        sys.stdout.write(json.dumps({"ok": False, "error": tb}) + "\n")
        sys.stdout.flush()
"#;

static WARM_WORKERS_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enables or disables persistent warm Python script workers (Settings › Environment).
pub fn set_warm_workers_enabled(on: bool) {
    WARM_WORKERS_ENABLED.store(on, Ordering::Relaxed);
    if !on {
        restart_python_workers();
    }
}

pub fn warm_workers_enabled() -> bool {
    WARM_WORKERS_ENABLED.load(Ordering::Relaxed)
}

fn cache_base() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("lucem")
}

/// Cache directory where Lucem creates managed `uv` `.venv` environments per dependency set.
pub fn managed_venvs_dir() -> PathBuf {
    cache_base().join("venvs")
}

fn run_checked(cmd: &mut Command, what: &str) -> Result<(), String> {
    let out = cmd.output().map_err(|e| format!("{what} could not start: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
        Err(format!("{what} failed:\n{}", tail.join("\n")))
    }
}

/// Ensures a cached `uv`-managed `.venv` exists for `deps` (plus `numpy`), precompiling bytecode
/// (`--compile-bytecode`) so startup is fast. Returns the venv's Python executable.
pub fn ensure_managed_venv(deps: &[String]) -> Result<PathBuf, String> {
    let mut all = vec!["numpy".to_string()];
    all.extend(deps.iter().cloned());
    all.sort();
    all.dedup();
    let mut h = DefaultHasher::new();
    all.hash(&mut h);
    let dir = managed_venvs_dir().join(format!("{:016x}", h.finish()));
    let py = if cfg!(windows) { dir.join("Scripts").join("python.exe") } else { dir.join("bin").join("python") };
    if py.exists() {
        return Ok(py);
    }
    let uv = crate::data::tools::program(crate::data::tools::Tool::Uv);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    run_checked(Command::new(&uv).args(["venv", "--quiet"]).arg(&dir), "uv venv")?;
    let mut install = Command::new(&uv);
    install.args(["pip", "install", "--quiet", "--compile-bytecode", "--python"]).arg(&py);
    for pkg in &all {
        install.arg(pkg);
    }
    run_checked(&mut install, "uv pip install")?;
    Ok(py)
}

struct WarmWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl WarmWorker {
    fn spawn(mut cmd: Command) -> Result<Self, String> {
        let scripts_cache = cache_base().join("scripts");
        std::fs::create_dir_all(&scripts_cache).map_err(|e| e.to_string())?;
        let worker_path = scripts_cache.join("warm_worker.py");
        std::fs::write(&worker_path, PYTHON_WARM_WORKER).map_err(|e| e.to_string())?;
        let mut child = cmd
            .arg(&worker_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("warm python worker could not start: {e}"))?;
        let stdin = child.stdin.take().ok_or("worker stdin unavailable")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("worker stdout unavailable")?);
        Ok(Self { child, stdin, stdout })
    }

    fn run(&mut self, script: &Path, folder: &Path) -> Result<(), String> {
        let req = serde_json::json!({
            "script": script.to_string_lossy(),
            "folder": folder.to_string_lossy(),
        });
        writeln!(self.stdin, "{req}").map_err(|e| format!("worker write failed: {e}"))?;
        self.stdin.flush().map_err(|e| format!("worker flush failed: {e}"))?;
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).map_err(|e| format!("worker read failed: {e}"))?;
        if n == 0 {
            return Err("warm python worker exited unexpectedly".into());
        }
        let resp: serde_json::Value = serde_json::from_str(line.trim()).map_err(|e| format!("invalid worker reply: {e}"))?;
        if resp.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            Ok(())
        } else {
            let err = resp.get("error").and_then(|v| v.as_str()).unwrap_or("script failed");
            let tail: Vec<&str> = err.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
            Err(tail.join("\n"))
        }
    }
}

impl Drop for WarmWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn warm_workers() -> &'static Mutex<HashMap<String, WarmWorker>> {
    static WORKERS: OnceLock<Mutex<HashMap<String, WarmWorker>>> = OnceLock::new();
    WORKERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Number of currently warm persistent Python worker processes.
pub fn warm_worker_count() -> usize {
    crate::platform::lock(warm_workers()).len()
}

/// Stops all warm Python workers (the next script run will spawn a fresh one).
pub fn restart_python_workers() {
    crate::platform::lock(warm_workers()).clear();
}

/// Runs `script` in `folder` inside the warm worker keyed by `env_key`, spawning or transparently
/// respawning once if the worker died.
pub fn run_in_warm_worker(env_key: &str, make_cmd: &dyn Fn() -> Command, script: &Path, folder: &Path) -> Result<(), String> {
    let mut pool = crate::platform::lock(warm_workers());
    let run_once = |pool: &mut HashMap<String, WarmWorker>| -> Result<(), String> {
        if !pool.contains_key(env_key) {
            pool.insert(env_key.to_string(), WarmWorker::spawn(make_cmd())?);
        }
        pool.get_mut(env_key).unwrap().run(script, folder)
    };
    if let Err(first_err) = run_once(&mut pool) {
        if first_err.contains("exited unexpectedly") || first_err.contains("write failed") {
            pool.remove(env_key);
            run_once(&mut pool)?;
        } else {
            return Err(first_err);
        }
    }
    Ok(())
}
