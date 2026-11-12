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

impl PluginProcess {
    /// Starts `program serve`.
    pub fn start(program: &str) -> Result<Self, String> {
        let mut child = Command::new(program).arg("serve").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e| format!("{program}: {e}"))?;
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
