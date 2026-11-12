//! `lucem ops`: every pipeline operation, as people and programs read it. Each comes with its
//! family, its input and output types, its parameters with their defaults, and what it does (the
//! operation's own documentation). The documentation's operations page is made from
//! `lucem ops --json`, and an agent finds the operations there.

use serde_json::Value;

use crate::Args;

/// The operations as JSON: `[{op, title, family, inputs, output, params, about}]` (shared with
/// the agent's `list_operations`).
fn as_json() -> Value {
    lucem_mcp::work::operations()
}

pub fn ops(mut args: Args) -> Result<(), String> {
    use std::io::Write as _;
    let json = args.take_flag("--json");
    let list = as_json();
    // Written through a lock, ending quietly when the reader stops (`lucem ops | head`)
    let mut out = std::io::stdout().lock();
    if json {
        let _ = writeln!(out, "{}", serde_json::to_string_pretty(&list).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let mut family = "";
    for o in list.as_array().into_iter().flatten() {
        let f = o["family"].as_str().unwrap_or("");
        if f != family {
            if writeln!(out, "\n{f}").is_err() {
                return Ok(());
            }
            family = f;
        }
        let ins: Vec<&str> = o["inputs"].as_array().into_iter().flatten().filter_map(|i| i["type"].as_str()).collect();
        if writeln!(out, "  {:<24} {:<24} {} → {}", o["op"].as_str().unwrap_or(""), o["title"].as_str().unwrap_or(""), ins.join(" + "), o["output"].as_str().unwrap_or("")).is_err() {
            return Ok(());
        }
    }
    Ok(())
}
