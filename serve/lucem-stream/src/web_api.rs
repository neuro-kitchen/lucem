//! Same-origin JSON API for `lucem serve --web` so paired browser pages can:
//! - inspect and select the server's Python/`.venv`, `uv`, Rust toolchain, and discovered plugin ops
//!   (`GET /api/env`, `POST /api/env/python`),
//! - create, read, and save `.py` / `.rs` script nodes on the server (`POST /api/scripts/new`,
//!   `GET /api/scripts/read`, `POST /api/scripts/write`),
//! - browse and inspect tabular measurement outputs in the browser's Analyze workspace
//!   (`GET /api/analyze/tables`, `GET /api/analyze/table`).
//!
//! Standalone GitHub Pages builds (`runs_alone() == true`) have no server and never hit these routes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

use http_body_util::{BodyExt as _, Full};
use hyper::body::{Bytes, Incoming};
use hyper::{Method, Response, StatusCode};
use serde_json::{json, Value};

type Reply = Response<Full<Bytes>>;

static PYTHON_OVERRIDE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn python_override() -> &'static Mutex<Option<String>> {
    PYTHON_OVERRIDE.get_or_init(|| Mutex::new(std::env::var("LUCEM_PYTHON").ok().filter(|s| !s.trim().is_empty())))
}

fn json_reply(status: StatusCode, body: Value) -> Reply {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("cache-control", "no-store")
        .body(Full::new(Bytes::from(body.to_string())))
        .expect("json response")
}

fn config_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(x).join("lucem");
    }
    if let Some(h) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(h).join(".config").join("lucem");
    }
    std::env::temp_dir().join("lucem")
}

fn plugin_ops_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(env_dirs) = std::env::var("LUCEM_PLUGINS") {
        for p in std::env::split_paths(&env_dirs) {
            dirs.push(p.join("ops"));
        }
    }
    dirs.push(config_dir().join("plugins").join("ops"));
    dirs
}

fn scripts_dir() -> PathBuf {
    config_dir().join("scripts")
}

fn allowed_script_path(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "py" && ext != "rs" {
        return false;
    }
    let canon = path.canonicalize().ok().or_else(|| {
        let parent = path.parent()?.canonicalize().ok()?;
        Some(parent.join(path.file_name()?))
    });
    let Some(canon) = canon else { return false };
    let mut roots = plugin_ops_dirs();
    roots.push(scripts_dir());
    roots.into_iter().filter_map(|r| r.canonicalize().ok()).any(|r| canon.starts_with(r))
}

fn cmd_version(prog: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(prog).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next().map(str::trim).filter(|l| !l.is_empty());
    line.map(str::to_string).or_else(|| {
        let e = String::from_utf8_lossy(&out.stderr);
        e.lines().next().map(|l| l.trim().to_string()).filter(|l| !l.is_empty())
    })
}

fn parse_script_header(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let comment = if path.extension().and_then(|e| e.to_str()) == Some("rs") { "//" } else { "#" };
    let mut name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("script").to_string();
    let mut family = "Custom".to_string();
    let mut description = String::new();
    let mut halo: usize = 0;
    for line in text.lines().take(60) {
        let t = line.trim();
        let Some(rest) = t.strip_prefix(comment).map(str::trim) else { continue };
        let Some(spec) = rest.strip_prefix("lucem:").map(str::trim) else { continue };
        if let Some((k, v)) = spec.split_once('=') {
            let v = v.trim().trim_matches('"').trim_matches('\'');
            match k.trim() {
                "name" if !v.is_empty() => name = v.to_string(),
                "family" if !v.is_empty() => family = v.to_string(),
                "description" => description = v.to_string(),
                "halo" => halo = v.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
    json!({
        "name": name,
        "family": family,
        "description": description,
        "halo": halo,
        "path": path.to_string_lossy(),
        "lang": path.extension().and_then(|e| e.to_str()).unwrap_or("py"),
    })
}

fn discover_plugin_ops() -> Vec<Value> {
    let mut out = Vec::new();
    for dir in plugin_ops_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if p.is_file() && (ext == "py" || ext == "rs") {
                out.push(parse_script_header(&p));
            }
        }
    }
    out
}

fn env_snapshot() -> Value {
    let override_path = python_override().lock().unwrap_or_else(|e| e.into_inner()).clone();
    let py_prog = override_path.as_deref().unwrap_or("python3");
    let py_ver = cmd_version(py_prog, &["--version"]).or_else(|| cmd_version("python", &["--version"]));
    let uv_ver = cmd_version("uv", &["--version"]);
    let rust_ver = cmd_version("rustc", &["--version"]);
    let plugins = discover_plugin_ops();
    let plugins_dir = plugin_ops_dirs().last().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    json!({
        "python": {
            "configured": override_path,
            "executable": py_prog,
            "version": py_ver,
        },
        "uv": { "version": uv_ver },
        "rust": { "version": rust_ver },
        "plugins_dir": plugins_dir,
        "scripts_dir": scripts_dir().to_string_lossy(),
        "plugin_ops": plugins,
    })
}

fn query_param(query: Option<&str>, key: &str) -> Option<String> {
    let q = query?;
    for pair in q.split('&') {
        if let Some((k, v)) = pair.split_once('=')
            && k == key
        {
            return Some(percent_decode(v));
        }
    }
    None
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn collect_tables(root: &Path) -> Vec<Value> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if !name.starts_with('.') && !name.ends_with(".zarr") {
                    stack.push(p);
                }
            } else if p.is_file() {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                if matches!(ext.as_str(), "csv" | "tsv") {
                    let rel = p.strip_prefix(root).unwrap_or(&p).to_string_lossy().into_owned();
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    found.push(json!({ "name": rel, "format": ext, "bytes": size }));
                }
            }
        }
        if found.len() >= 64 {
            break;
        }
    }
    found
}

fn read_delimited_table(root: &Path, rel: &str) -> Result<Value, String> {
    let full = root.join(rel);
    let canon_root = root.canonicalize().map_err(|e| e.to_string())?;
    let canon = full.canonicalize().map_err(|e| e.to_string())?;
    if !canon.starts_with(&canon_root) || !canon.is_file() {
        return Err("table path is outside the server data folder".into());
    }
    let sep = if canon.extension().and_then(|e| e.to_str()) == Some("tsv") { '\t' } else { ',' };
    let text = std::fs::read_to_string(&canon).map_err(|e| e.to_string())?;
    let mut lines = text.lines();
    let header_line = lines.next().unwrap_or("");
    let columns: Vec<String> = header_line.split(sep).map(|s| s.trim().trim_matches('"').to_string()).collect();
    let mut rows = Vec::new();
    let mut total_rows = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        total_rows += 1;
        if rows.len() < 500 {
            rows.push(line.split(sep).map(|s| s.trim().trim_matches('"').to_string()).collect::<Vec<_>>());
        }
    }
    Ok(json!({
        "name": rel,
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
    }))
}

fn check_script_diagnostics(lang: &str, content: &str) -> Vec<Value> {
    if lang != "py" {
        return Vec::new();
    }
    let override_path = python_override().lock().unwrap_or_else(|e| e.into_inner()).clone();
    let py = override_path.as_deref().unwrap_or("python3");
    let checker = "import ast, json, sys\nsrc = sys.stdin.read()\ntry:\n    ast.parse(src)\n    print('[]')\nexcept SyntaxError as e:\n    print(json.dumps([{'line': e.lineno or 1, 'message': e.msg}]))\n";
    let Ok(mut child) = Command::new(py)
        .args(["-c", checker])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return Vec::new();
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write as _;
        let _ = stdin.write_all(content.as_bytes());
    }
    let Ok(out) = child.wait_with_output() else { return Vec::new() };
    serde_json::from_slice::<Vec<Value>>(&out.stdout).unwrap_or_default()
}

pub async fn handle(request: hyper::Request<Incoming>, data_root: &Path) -> Reply {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let query = request.uri().query().map(str::to_string);

    match (method, path.as_str()) {
        (Method::GET, "/api/env") => {
            let snap = tokio::task::spawn_blocking(env_snapshot).await.unwrap_or_else(|_| json!({}));
            json_reply(StatusCode::OK, snap)
        }
        (Method::POST, "/api/env/python") => {
            let body = match request.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(e) => return json_reply(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
            };
            let req: Value = serde_json::from_slice(&body).unwrap_or_default();
            let new_path = req.get("path").and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
            *python_override().lock().unwrap_or_else(|e| e.into_inner()) = new_path;
            let snap = tokio::task::spawn_blocking(env_snapshot).await.unwrap_or_else(|_| json!({}));
            json_reply(StatusCode::OK, snap)
        }
        (Method::GET, "/api/scripts/read") => {
            let Some(p) = query_param(query.as_deref(), "path") else {
                return json_reply(StatusCode::BAD_REQUEST, json!({"error": "missing path"}));
            };
            let pb = PathBuf::from(&p);
            if !allowed_script_path(&pb) {
                return json_reply(StatusCode::FORBIDDEN, json!({"error": "script path not in plugins/ops or scripts folder"}));
            }
            match std::fs::read_to_string(&pb) {
                Ok(content) => json_reply(StatusCode::OK, json!({"path": p, "content": content})),
                Err(e) => json_reply(StatusCode::NOT_FOUND, json!({"error": e.to_string()})),
            }
        }
        (Method::POST, "/api/scripts/write") => {
            let body = match request.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(e) => return json_reply(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
            };
            let req: Value = serde_json::from_slice(&body).unwrap_or_default();
            let Some(p) = req.get("path").and_then(|v| v.as_str()) else {
                return json_reply(StatusCode::BAD_REQUEST, json!({"error": "missing path"}));
            };
            let Some(content) = req.get("content").and_then(|v| v.as_str()) else {
                return json_reply(StatusCode::BAD_REQUEST, json!({"error": "missing content"}));
            };
            let pb = PathBuf::from(p);
            if !allowed_script_path(&pb) {
                return json_reply(StatusCode::FORBIDDEN, json!({"error": "script path not in plugins/ops or scripts folder"}));
            }
            match std::fs::write(&pb, content) {
                Ok(()) => json_reply(StatusCode::OK, json!({"ok": true, "path": p})),
                Err(e) => json_reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()})),
            }
        }
        (Method::POST, "/api/scripts/new") => {
            let body = match request.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(e) => return json_reply(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
            };
            let req: Value = serde_json::from_slice(&body).unwrap_or_default();
            let lang = req.get("lang").and_then(|v| v.as_str()).unwrap_or("py");
            let ext = if lang == "rs" { "rs" } else { "py" };
            let dir = scripts_dir();
            if let Err(e) = std::fs::create_dir_all(&dir) {
                return json_reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()}));
            }
            let mut idx = 1usize;
            let file_path = loop {
                let candidate = dir.join(format!("step_{idx}.{ext}"));
                if !candidate.exists() {
                    break candidate;
                }
                idx += 1;
            };
            let starter = if ext == "rs" {
                "// lucem: name = \"Custom Rust Step\"\n// lucem: input image: image\n// lucem: param gain = 1.5\n// lucem: output image\n\npub fn run(inputs: &Inputs, params: &Params) -> Result<Array, String> {\n    let src = inputs.image(\"image\")?;\n    let gain = params.get(\"gain\") as f32;\n    let data = src.data.iter().map(|&v| (v * gain).clamp(0.0, 1.0)).collect();\n    Ok(Array::image(src.shape.clone(), data))\n}\n"
            } else {
                "# /// script\n# dependencies = [\"numpy\"]\n# ///\n# lucem: name = \"Custom Python Step\"\n# lucem: family = \"Filter\"\n# lucem: halo = 4\n# lucem: input image: image\n# lucem: param gain = 1.5\n# lucem: output image\n\nimport numpy as np\n\ndef run(image, gain):\n    return np.clip(image.astype(np.float32) * float(gain), 0.0, 1.0)\n"
            };
            if let Err(e) = std::fs::write(&file_path, starter) {
                return json_reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()}));
            }
            let header = parse_script_header(&file_path);
            json_reply(StatusCode::OK, json!({"path": file_path.to_string_lossy(), "content": starter, "header": header}))
        }
        (Method::POST, "/api/scripts/check") => {
            let body = match request.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(e) => return json_reply(StatusCode::BAD_REQUEST, json!({"error": e.to_string()})),
            };
            let req: Value = serde_json::from_slice(&body).unwrap_or_default();
            let lang = req.get("lang").and_then(|v| v.as_str()).unwrap_or("py").to_string();
            let content = req.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let diags = tokio::task::spawn_blocking(move || check_script_diagnostics(&lang, &content))
                .await
                .unwrap_or_default();
            json_reply(StatusCode::OK, json!({"diagnostics": diags}))
        }
        (Method::GET, "/api/analyze/tables") => {
            let root = data_root.to_path_buf();
            let tables = tokio::task::spawn_blocking(move || collect_tables(&root)).await.unwrap_or_default();
            json_reply(StatusCode::OK, json!({"tables": tables}))
        }
        (Method::GET, "/api/analyze/table") => {
            let Some(rel) = query_param(query.as_deref(), "name") else {
                return json_reply(StatusCode::BAD_REQUEST, json!({"error": "missing name"}));
            };
            let root = data_root.to_path_buf();
            match tokio::task::spawn_blocking(move || read_delimited_table(&root, &rel)).await {
                Ok(Ok(v)) => json_reply(StatusCode::OK, v),
                Ok(Err(e)) => json_reply(StatusCode::BAD_REQUEST, json!({"error": e})),
                Err(e) => json_reply(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": e.to_string()})),
            }
        }
        _ => json_reply(StatusCode::NOT_FOUND, json!({"error": "unknown api route"})),
    }
}
