//! Served images in a browser page: the page's own server (or, the docs viewer, its engine
//! worker on the files the user picked).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use lucem_io::Pyramid;
use lucem_stream::web_client::PageLink;
use lucem_stream::wire;

static LINK: OnceLock<Arc<PageLink>> = OnceLock::new();
static LISTED: Mutex<Option<Result<Vec<String>, String>>> = Mutex::new(None);

use crate::platform::lock;

/// The page's connection to its server, once it signed in.
pub fn set_page_link(link: Arc<PageLink>) {
    let _ = LINK.set(link);
}

/// The page's connection, if it signed in.
pub fn page_link() -> Option<Arc<PageLink>> {
    LINK.get().cloned()
}

/// Whether there is a server to ask (the page's, once signed in, or its own engine).
pub fn has_servers() -> bool {
    LINK.get().is_some()
}

/// Whether the page runs alone (the docs viewer): its images are the files the user picks.
pub fn runs_alone() -> bool {
    LINK.get().is_some_and(|l| l.transport() == lucem_stream::web_client::Transport::Worker)
}

/// Asks the user for a project (`project`: its place, or an OME-Zarr, with the browser's folder
/// chooser, where it asks once to let the page read it) or a file (the file chooser, as for a
/// PDF: no question), and hands what was picked to the page's engine: the loader's pickers
/// (`window.lucemPickFolder` / `lucemPickFiles`) list names first and give lazy `File`s (nothing
/// is read until a tile needs it). The images among them, by their served names; none when the
/// dialog was closed.
pub async fn pick_files(project: bool) -> Result<Vec<String>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let name = if project { "lucemPickFolder" } else { "lucemPickFiles" };
    let pick: js_sys::Function = js_sys::Reflect::get(&window, &name.into()).ok().and_then(|f| wasm_bindgen::JsCast::dyn_into(f).ok()).ok_or_else(|| format!("this page's loader has no {name}"))?;
    let promise: js_sys::Promise = wasm_bindgen::JsCast::dyn_into(pick.call0(&wasm_bindgen::JsValue::NULL).map_err(|e| format!("{e:?}"))?).map_err(|_| "the picker gave no promise".to_string())?;
    let picked = wasm_bindgen_futures::JsFuture::from(promise).await.map_err(|e| format!("{e:?}"))?;
    if picked.is_null() || picked.is_undefined() {
        return Ok(Vec::new());
    }
    take_files(js_sys::Array::from(&picked)).await
}

/// Hands `files` (`[{path, file}]`) to the page's engine; the images among them, by their served
/// names (`lucem://local/<path>`). None found is an error that says what was given.
pub async fn take_files(files: js_sys::Array) -> Result<Vec<String>, String> {
    let link = LINK.get().ok_or("the page has no engine")?.clone();
    let n = files.length();
    link.set_files(files).map_err(|e| e.to_string())?;
    let images: Vec<String> = link.list().await.map_err(|e| e.to_string())?.into_iter().filter(|p| !p.starts_with("samples/")).map(|p| lucem_stream::remote_name(link.host(), &p)).collect();
    if images.is_empty() {
        return Err(format!("No image among the {n} file(s): lucem reads OME-Zarr (.ome.zarr, .zarr), TIFF, PNG, JPEG, BMP and WebP"));
    }
    Ok(images)
}

/// What is dropped on the page, as the loader hands it over (`[{path, file}]`): the loader's
/// drop handler (`window.lucemDrop`) is set by the first call.
pub fn drops() -> async_channel::Receiver<js_sys::Array> {
    let (tx, rx) = async_channel::unbounded();
    let handler = wasm_bindgen::closure::Closure::<dyn FnMut(wasm_bindgen::JsValue)>::new(move |entries: wasm_bindgen::JsValue| {
        let _ = tx.try_send(js_sys::Array::from(&entries));
    });
    if let Some(window) = web_sys::window() {
        let _ = js_sys::Reflect::set(&window, &"lucemDrop".into(), handler.as_ref());
    }
    // The page's for its whole life
    handler.forget();
    rx
}

/// Opens `lucem://<host>/<path>` on the page's server (from a worker: it waits).
pub fn open(name: &str) -> Result<Pyramid, String> {
    let (_, path) = lucem_stream::split_remote(name).ok_or_else(|| format!("{name}: not a served image (lucem://host:port/path)"))?;
    let link = LINK.get().ok_or("the page is not connected to its server")?;
    link.open(path, None).map_err(|e| format!("{name}: {e}"))
}

/// Asks the page's server for its images (from a worker: it waits); what was found.
pub fn refresh() -> String {
    let Some(link) = LINK.get() else { return "The page is not connected to its server".into() };
    let found = match link.request(wire::request::Kind::List(wire::ListImages {})) {
        Ok(wire::response::Kind::Images(list)) => Ok(list.paths),
        Ok(_) => Err("unexpected answer to a listing".to_string()),
        Err(e) => Err(e.to_string()),
    };
    let said = match &found {
        Ok(images) => format!("{} image(s) on {}: the server menu lists them", images.len(), link.welcome().server_name),
        Err(e) => format!("The server did not list its images: {e}"),
    };
    *lock(&LISTED) = Some(found);
    said
}

/// What the page's server offered when last asked, by its address.
pub fn last_listed() -> BTreeMap<String, Result<Vec<String>, String>> {
    match (LINK.get(), lock(&LISTED).clone()) {
        (Some(link), Some(found)) => BTreeMap::from([(link.host().to_string(), found)]),
        _ => BTreeMap::new(),
    }
}
