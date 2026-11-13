//! A plugin in server mode: `<program> serve`, kept alive so its model stays loaded. Blocking:
//! the app calls it off the UI thread.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::protocol::{Event, Request};

pub struct PluginProcess {
    pub program: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// The compute runtime plugins are started on (`LUCEM_RUNTIME` for them), when the program that
/// starts them has chosen one ([`set_plugin_runtime`]); otherwise they inherit the environment.
static PLUGIN_RUNTIME: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Plugins started from now on run on `runtime` (`cuda`, `wgpu`, `cpu`): the app sets its own, so
/// a model never opens a second GPU runtime beside the app's (two runtimes on one GPU share its
/// memory, and the second runs out of it).
pub fn set_plugin_runtime(runtime: Option<&str>) {
    if let Ok(mut r) = PLUGIN_RUNTIME.lock() {
        *r = runtime.map(str::to_string);
    }
}

impl PluginProcess {
    /// Starts `program serve` (on the runtime [`set_plugin_runtime`] chose, if any).
    pub fn start(program: &str) -> Result<Self, String> {
        let mut cmd = Command::new(program);
        cmd.arg("serve").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        if let Some(r) = PLUGIN_RUNTIME.lock().ok().and_then(|r| r.clone()) {
            cmd.env("LUCEM_RUNTIME", r);
        }
        let mut child = cmd.spawn().map_err(|e| format!("{program}: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
        Ok(Self { program: program.to_string(), child, stdin, stdout })
    }

    /// Whether the program is still running.
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Sends a request and reads events until its answer; `progress` sees the progress events.
    pub fn request(&mut self, req: &Request, progress: &mut dyn FnMut(&str, Option<f32>)) -> Result<serde_json::Value, String> {
        let line = serde_json::to_string(req).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{line}").and_then(|_| self.stdin.flush()).map_err(|e| format!("{}: {e}", self.program))?;
        loop {
            let mut buf = String::new();
            let n = self.stdout.read_line(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err(format!("{} stopped (see its messages in the terminal)", self.program));
            }
            let Ok(event) = serde_json::from_str::<Event>(buf.trim()) else { continue };
            match event {
                Event::Progress { stage, fraction } => progress(&stage, fraction),
                Event::Done { result } => return Ok(result),
                Event::Error { message } => return Err(message),
            }
        }
    }
}

impl Drop for PluginProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{}", serde_json::to_string(&Request::Quit).unwrap_or_default());
        let _ = self.stdin.flush();
        // A server busy with a long request is not waited for
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
