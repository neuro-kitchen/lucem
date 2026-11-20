//! Browser-side client for `lucem serve --web`'s same-origin `/api/*` endpoints.
//!
//! Only used on `wasm32` when connected to a `lucem serve` host (`!crate::data::remote::runs_alone()`).

#[cfg(target_family = "wasm")]
use serde::{Deserialize, Serialize};
#[cfg(target_family = "wasm")]
use wasm_bindgen::JsCast as _;
#[cfg(target_family = "wasm")]
use wasm_bindgen_futures::JsFuture;

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerPluginOp {
    pub name: String,
    pub family: String,
    pub description: String,
    pub halo: usize,
    pub path: String,
    pub lang: String,
}

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerPythonInfo {
    pub configured: Option<String>,
    pub executable: String,
    pub version: Option<String>,
}

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerToolVer {
    pub version: Option<String>,
}

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerEnvInfo {
    pub python: ServerPythonInfo,
    pub uv: ServerToolVer,
    pub rust: ServerToolVer,
    pub plugins_dir: String,
    pub scripts_dir: String,
    pub plugin_ops: Vec<ServerPluginOp>,
}

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerTableSummary {
    pub name: String,
    pub format: String,
    pub bytes: u64,
}

#[cfg(target_family = "wasm")]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerTableData {
    pub name: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub total_rows: usize,
}

#[cfg(target_family = "wasm")]
async fn fetch_json(url: &str, method: &str, body: Option<String>) -> Result<serde_json::Value, String> {
    let win = web_sys::window().ok_or("no browser window")?;
    let opts = web_sys::RequestInit::new();
    opts.set_method(method);
    if let Some(body_str) = body {
        let headers = web_sys::Headers::new().map_err(|e| format!("{e:?}"))?;
        headers.set("content-type", "application/json").map_err(|e| format!("{e:?}"))?;
        opts.set_headers(&headers);
        opts.set_body(&wasm_bindgen::JsValue::from_str(&body_str));
    }
    let req = web_sys::Request::new_with_str_and_init(url, &opts).map_err(|e| format!("{e:?}"))?;
    let resp_val = JsFuture::from(win.fetch_with_request(&req)).await.map_err(|e| format!("fetch failed: {e:?}"))?;
    let resp: web_sys::Response = resp_val.dyn_into().map_err(|_| "invalid Response")?;
    let text_val = JsFuture::from(resp.text().map_err(|e| format!("{e:?}"))?).await.map_err(|e| format!("{e:?}"))?;
    let text = text_val.as_string().unwrap_or_default();
    let val: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("invalid JSON: {e}"))?;
    if !resp.ok() {
        let msg = val.get("error").and_then(|v| v.as_str()).unwrap_or("server request failed");
        return Err(msg.to_string());
    }
    Ok(val)
}

#[cfg(target_family = "wasm")]
pub async fn fetch_env() -> Result<ServerEnvInfo, String> {
    let v = fetch_json("/api/env", "GET", None).await?;
    serde_json::from_value(v).map_err(|e| e.to_string())
}

#[cfg(target_family = "wasm")]
pub async fn set_server_python(path: Option<String>) -> Result<ServerEnvInfo, String> {
    let body = serde_json::json!({ "path": path }).to_string();
    let v = fetch_json("/api/env/python", "POST", Some(body)).await?;
    serde_json::from_value(v).map_err(|e| e.to_string())
}

#[cfg(target_family = "wasm")]
pub async fn new_server_script(lang: &str) -> Result<(String, String), String> {
    let body = serde_json::json!({ "lang": lang }).to_string();
    let v = fetch_json("/api/scripts/new", "POST", Some(body)).await?;
    let path = v.get("path").and_then(|s| s.as_str()).ok_or("missing path")?.to_string();
    let content = v.get("content").and_then(|s| s.as_str()).unwrap_or("").to_string();
    Ok((path, content))
}

#[cfg(target_family = "wasm")]
pub async fn read_server_script(path: &str) -> Result<String, String> {
    let encoded = js_sys::encode_uri_component(path).as_string().unwrap_or_else(|| path.to_string());
    let v = fetch_json(&format!("/api/scripts/read?path={encoded}"), "GET", None).await?;
    Ok(v.get("content").and_then(|s| s.as_str()).unwrap_or("").to_string())
}

#[cfg(target_family = "wasm")]
pub async fn write_server_script(path: &str, content: &str) -> Result<(), String> {
    let body = serde_json::json!({ "path": path, "content": content }).to_string();
    let _ = fetch_json("/api/scripts/write", "POST", Some(body)).await?;
    Ok(())
}

#[cfg(target_family = "wasm")]
pub async fn check_server_script(lang: &str, content: &str) -> Result<Vec<(usize, String)>, String> {
    let body = serde_json::json!({ "lang": lang, "content": content }).to_string();
    let v = fetch_json("/api/scripts/check", "POST", Some(body)).await?;
    let mut out = Vec::new();
    if let Some(arr) = v.get("diagnostics").and_then(|d| d.as_array()) {
        for item in arr {
            let line = item.get("line").and_then(|l| l.as_u64()).unwrap_or(1) as usize;
            let msg = item.get("message").and_then(|m| m.as_str()).unwrap_or("syntax error").to_string();
            out.push((line, msg));
        }
    }
    Ok(out)
}

#[cfg(target_family = "wasm")]
pub async fn list_server_tables() -> Result<Vec<ServerTableSummary>, String> {
    let v = fetch_json("/api/analyze/tables", "GET", None).await?;
    let list = v.get("tables").cloned().unwrap_or(serde_json::Value::Array(vec![]));
    serde_json::from_value(list).map_err(|e| e.to_string())
}

#[cfg(target_family = "wasm")]
pub async fn fetch_server_table(name: &str) -> Result<ServerTableData, String> {
    let encoded = js_sys::encode_uri_component(name).as_string().unwrap_or_else(|| name.to_string());
    let v = fetch_json(&format!("/api/analyze/table?name={encoded}"), "GET", None).await?;
    serde_json::from_value(v).map_err(|e| e.to_string())
}

#[cfg(target_family = "wasm")]
pub async fn fetch_ml_plugins() -> Result<(Vec<lucem_ml::PluginInfo>, Vec<String>), String> {
    let v = fetch_json("/api/ml/plugins", "GET", None).await?;
    let plugins = serde_json::from_value(v.get("plugins").cloned().unwrap_or_default()).map_err(|e| e.to_string())?;
    let problems = serde_json::from_value(v.get("problems").cloned().unwrap_or_default()).unwrap_or_default();
    Ok((plugins, problems))
}

#[cfg(target_family = "wasm")]
pub async fn install_server_model(program: &str, model: &str) -> Result<(), String> {
    let body = serde_json::json!({ "program": program, "model": model }).to_string();
    fetch_json("/api/ml/install", "POST", Some(body)).await.map(|_| ())
}

#[cfg(target_family = "wasm")]
pub async fn save_server_recipe(recipe: &crate::data::recipes::Recipe) -> Result<String, String> {
    let body = serde_json::to_string(recipe).map_err(|e| e.to_string())?;
    let v = fetch_json("/api/recipes/save", "POST", Some(body)).await?;
    Ok(v.get("path").and_then(|s| s.as_str()).unwrap_or("saved").to_string())
}
