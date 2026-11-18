//! A browser page's connection to the server that served it (`lucem serve --web`): WebTransport
//! first (one QUIC stream per request: a slow tile holds nothing up, and closing a stream drops
//! a tile no longer wanted), a WebSocket where UDP does not get through. Both carry the same
//! protobuf requests ([`crate::wire`]) after the page says who it is ([`crate::pair`]).
//!
//! - **WebTransport** is driven by a few lines of JavaScript in the page's loader
//!   (`window.lucemTransport`, see `app/lucem-app-web/index.html`): `open(url, hash)` gives a
//!   session with `control.send(text)`, `control.next()` and `request(bytes)` (a stream per
//!   call). That keeps this crate off `web-sys`'s unstable WebTransport bindings.
//! - **Threads.** The connection belongs to the page's main thread, which never waits: a
//!   request made there is awaited ([`PageLink::request_async`]); one made on a worker (a tile
//!   being read for the canvas) waits for its answer ([`PageLink::request`]), which only a
//!   build with threads can do. (As dsp-kitchen's `web_client`.)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use futures_util::StreamExt as _;
use lucem_io::{Pyramid, TileSource};
use prost::Message as _;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast as _, JsValue};

use crate::error::{StreamError, StreamResult};
use crate::pair::{ClientPairing, PairReply, ServeInfo, WebHello, WebWelcome, WS_PATH, WT_PATH};
use crate::wire;

type Answer = Result<wire::Response, String>;

const CLOSED: &str = "the connection to the server closed";

/// How the page says who it is.
pub enum Hello {
    /// A token from an earlier pairing.
    Token(String),
    /// A pairing code, and the name the server will know this page by.
    Code { code: String, name: String },
}

/// Which transport the page got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    WebTransport,
    WebSocket,
}

/// Locks without waiting on the page's main thread (which may not wait): held for an instant.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    loop {
        match mutex.try_lock() {
            Ok(guard) => return guard,
            Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => std::hint::spin_loop(),
        }
    }
}

#[derive(Default)]
struct Slot {
    answer: Mutex<Option<Answer>>,
    arrived: std::sync::Condvar,
}

enum Waiter {
    Task(futures_channel::oneshot::Sender<Answer>),
    Worker(Arc<Slot>),
}

impl Waiter {
    fn deliver(self, answer: Answer) {
        match self {
            Waiter::Task(tx) => {
                let _ = tx.send(answer);
            }
            Waiter::Worker(slot) => {
                *lock(&slot.answer) = Some(answer);
                slot.arrived.notify_all();
            }
        }
    }
}

/// The page's connection, once the server let it in.
pub struct PageLink {
    /// Requests for the main thread to send: an id and the encoded request.
    out: futures_channel::mpsc::UnboundedSender<(u32, Vec<u8>)>,
    pending: Mutex<HashMap<u32, Waiter>>,
    next: AtomicU32,
    welcome: WebWelcome,
    transport: Transport,
    /// The server's address as the page reaches it (`host:port`), for `lucem://` names.
    host: String,
}

fn kind_of(answer: Answer) -> StreamResult<wire::response::Kind> {
    match answer.map_err(StreamError::protocol)?.kind {
        Some(wire::response::Kind::Error(e)) => Err(StreamError::Server(e)),
        Some(kind) => Ok(kind),
        None => Err(StreamError::protocol("the server sent an empty answer")),
    }
}

impl PageLink {
    /// Who the server says the page is.
    pub fn welcome(&self) -> &WebWelcome {
        &self.welcome
    }

    pub fn transport(&self) -> Transport {
        self.transport
    }

    /// The server's `host:port`.
    pub fn host(&self) -> &str {
        &self.host
    }

    fn send(&self, kind: wire::request::Kind, waiter: Waiter) -> StreamResult<()> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.pending).insert(id, waiter);
        self.out.unbounded_send((id, wire::Request { kind: Some(kind) }.encode_to_vec())).map_err(|_| StreamError::protocol(CLOSED))
    }

    fn deliver(&self, id: u32, answer: Answer) {
        let waiting = lock(&self.pending).remove(&id);
        if let Some(w) = waiting {
            w.deliver(answer);
        }
    }

    /// Asks the server without waiting (the page's main thread, or any task).
    pub async fn request_async(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.send(kind, Waiter::Task(tx))?;
        kind_of(rx.await.map_err(|_| StreamError::protocol(CLOSED))?)
    }

    /// Waits for the answer: from a worker of a build with threads only (the page's main thread
    /// may not wait).
    pub fn request(&self, kind: wire::request::Kind) -> StreamResult<wire::response::Kind> {
        if cfg!(not(target_feature = "atomics")) {
            return Err(StreamError::protocol("an image on the server is read from a worker: this build of the page has none"));
        }
        let slot = Arc::new(Slot::default());
        self.send(kind, Waiter::Worker(slot.clone()))?;
        let mut answer = slot.answer.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(answer) = answer.take() {
                return kind_of(answer);
            }
            answer = slot.arrived.wait(answer).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Everyone waiting learns the connection is gone.
    fn closed(&self) {
        let waiting: Vec<Waiter> = lock(&self.pending).drain().map(|(_, w)| w).collect();
        for w in waiting {
            w.deliver(Err(CLOSED.into()));
        }
        self.out.close_channel();
    }

    /// The images the server offers (awaited).
    pub async fn list(&self) -> StreamResult<Vec<String>> {
        match self.request_async(wire::request::Kind::List(wire::ListImages {})).await? {
            wire::response::Kind::Images(list) => Ok(list.paths),
            _ => Err(StreamError::protocol("unexpected answer to a listing")),
        }
    }

    /// Calls `method` of the server's plugin `plugin` (awaited): its JSON reply.
    pub async fn plugin_call(&self, plugin: &str, method: &str, body: &serde_json::Value) -> StreamResult<serde_json::Value> {
        let call = wire::PluginCall { plugin: plugin.into(), method: method.into(), body: serde_json::to_vec(body)? };
        match self.request_async(wire::request::Kind::Plugin(call)).await? {
            wire::response::Kind::Plugin(reply) if reply.body.is_empty() => Ok(serde_json::Value::Null),
            wire::response::Kind::Plugin(reply) => Ok(serde_json::from_slice(&reply.body)?),
            _ => Err(StreamError::protocol("unexpected answer to a plugin call")),
        }
    }

    /// Opens the served image `path` as a pyramid (from a worker: it waits for the description),
    /// its tiles read over this connection. `processing`: a pipeline (`PipelineSpec` JSON) the
    /// server runs on every tile.
    pub fn open(self: &Arc<Self>, path: &str, processing: Option<String>) -> StreamResult<Pyramid> {
        let info = match self.request(wire::request::Kind::Info(wire::GetInfo { path: path.into() }))? {
            wire::response::Kind::Info(info) => wire::info_from_wire(&info),
            _ => return Err(StreamError::protocol("unexpected answer to an image's description")),
        };
        if info.levels.is_empty() {
            return Err(StreamError::protocol("the image has no levels"));
        }
        let (levels, channels) = crate::protocol::pyramid_parts(&info);
        let name = crate::protocol::remote_name(&self.host, path);
        let source = Arc::new(PageTiles { link: self.clone(), path: path.into(), processing });
        Pyramid::from_source(info.name.clone(), Some(PathBuf::from(name)), levels, channels, info.pixel_size_um, info.z_spacing_um, info.dtype.clone(), source).map_err(|e| StreamError::protocol(e.to_string()))
    }
}

/// A served image's tiles, read over the page's connection.
struct PageTiles {
    link: Arc<PageLink>,
    path: String,
    processing: Option<String>,
}

impl TileSource for PageTiles {
    fn tile(&self, level: usize, channel: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let ask = wire::GetTile { path: self.path.clone(), level: level as u32, channel: channel as u32, z: z as u32, tx: tx as u32, ty: ty as u32, processing: self.processing.clone().map(|spec_json| wire::Processing { spec_json }) };
        match self.link.request(wire::request::Kind::Tile(ask)).map_err(|e| e.to_string())? {
            wire::response::Kind::Tile(t) => wire::decode_tile(&t),
            _ => Err("unexpected answer to a tile".into()),
        }
    }
}

// ---- connecting ---------------------------------------------------------------------------------

fn js(e: JsValue) -> StreamError {
    StreamError::protocol(format!("{e:?}"))
}

/// Calls `object.name(args…)` and awaits the promise it returns.
async fn call(object: &JsValue, name: &str, args: &[JsValue]) -> StreamResult<JsValue> {
    let f: js_sys::Function = js_sys::Reflect::get(object, &name.into()).map_err(js)?.dyn_into().map_err(|_| StreamError::protocol(format!("{name} is not a function")))?;
    let array = js_sys::Array::new();
    for a in args {
        array.push(a);
    }
    let promise: js_sys::Promise = f.apply(object, &array).map_err(js)?.dyn_into().map_err(|_| StreamError::protocol(format!("{name} gave no promise")))?;
    wasm_bindgen_futures::JsFuture::from(promise).await.map_err(js)
}

/// How the page and the server exchange JSON while the page says who it is.
enum Control {
    WebTransport(JsValue),
    WebSocket { socket: web_sys::WebSocket, texts: futures_channel::mpsc::UnboundedReceiver<Option<String>> },
}

impl Control {
    async fn send(&mut self, text: String) -> StreamResult<()> {
        match self {
            Control::WebTransport(session) => {
                let control = js_sys::Reflect::get(session, &"control".into()).map_err(js)?;
                call(&control, "send", &[text.into()]).await.map(|_| ())
            }
            Control::WebSocket { socket, .. } => socket.send_with_str(&text).map_err(js),
        }
    }

    async fn next(&mut self) -> StreamResult<String> {
        match self {
            Control::WebTransport(session) => {
                let control = js_sys::Reflect::get(session, &"control".into()).map_err(js)?;
                call(&control, "next", &[]).await?.as_string().ok_or_else(|| StreamError::protocol("the server closed the connection"))
            }
            Control::WebSocket { texts, .. } => texts.next().await.flatten().ok_or_else(|| StreamError::protocol("the server closed the connection")),
        }
    }
}

/// Says who the page is on `control`: the welcome, and the new token when the page paired.
async fn hello(control: &mut Control, hello: Hello) -> StreamResult<(WebWelcome, Option<String>)> {
    let json = |e: serde_json::Error| StreamError::protocol(e.to_string());
    let mut token = None;
    match hello {
        Hello::Token(t) => control.send(serde_json::to_string(&WebHello::Token { token: t }).map_err(json)?).await?,
        Hello::Code { code, name } => {
            let (pairing, hello) = ClientPairing::start(&code, &name)?;
            control.send(serde_json::to_string(&WebHello::Pair { pair: hello }).map_err(json)?).await?;
            let reply: PairReply = serde_json::from_str(&control.next().await?).map_err(json)?;
            let (pairing, proof) = pairing.token(reply)?;
            control.send(serde_json::to_string(&proof).map_err(json)?).await?;
            let reply: PairReply = serde_json::from_str(&control.next().await?).map_err(json)?;
            token = Some(pairing.finish(reply)?.token);
        }
    }
    let welcome: WebWelcome = serde_json::from_str(&control.next().await.map_err(|_| StreamError::protocol("the server did not let the page in"))?).map_err(json)?;
    Ok((welcome, token))
}

/// The page's `host:port` and whether it was loaded over HTTPS.
fn page_host() -> StreamResult<(String, bool)> {
    let location = web_sys::window().ok_or_else(|| StreamError::protocol("no window"))?.location();
    Ok((location.host().map_err(js)?, location.protocol().map_err(js)? == "https:"))
}

/// Connects to the server that served the page (on its main thread): WebTransport when the
/// browser and the network allow it, else the WebSocket. Returns the connection, and the new
/// token when the page paired.
pub async fn connect(serve: &ServeInfo, how: impl Fn() -> Hello) -> StreamResult<(Arc<PageLink>, Option<String>)> {
    let (host, secure) = page_host()?;
    let wt = web_sys::window().and_then(|w| js_sys::Reflect::get(&w, &"lucemTransport".into()).ok()).filter(|t| !t.is_undefined());
    if let (Some(transport), Some(port)) = (wt, serve.webtransport_port) {
        let hostname = host.rsplit_once(':').map_or(host.as_str(), |(h, _)| h).to_string();
        let url = format!("https://{hostname}:{port}{WT_PATH}");
        let hash = js_sys::Uint8Array::from(serve.certificate_sha256.as_slice());
        match call(&transport, "open", &[url.into(), hash.into()]).await {
            Ok(session) => return over_webtransport(session, host, how()).await,
            Err(e) => log(&format!("lucem: WebTransport did not connect ({e}); the WebSocket instead")),
        }
    }
    let url = format!("{}://{host}{WS_PATH}", if secure { "wss" } else { "ws" });
    over_websocket(&url, host, how()).await
}

/// `console.log` (without web-sys's `console` feature).
fn log(text: &str) {
    let console = web_sys::window().and_then(|w| js_sys::Reflect::get(&w, &"console".into()).ok());
    if let Some(f) = console.as_ref().and_then(|c| js_sys::Reflect::get(c, &"log".into()).ok()).and_then(|f| f.dyn_into::<js_sys::Function>().ok()) {
        let _ = f.call1(console.as_ref().unwrap_or(&JsValue::NULL), &text.into());
    }
}

async fn over_webtransport(session: JsValue, host: String, how: Hello) -> StreamResult<(Arc<PageLink>, Option<String>)> {
    let mut control = Control::WebTransport(session.clone());
    let (welcome, token) = hello(&mut control, how).await?;
    let (out, mut requests) = futures_channel::mpsc::unbounded::<(u32, Vec<u8>)>();
    let link = Arc::new(PageLink { out, pending: Mutex::default(), next: AtomicU32::new(1), welcome, transport: Transport::WebTransport, host });
    // Each request on a stream of its own, answered as soon as it is (any order)
    let sender = link.clone();
    wasm_bindgen_futures::spawn_local(async move {
        while let Some((id, bytes)) = requests.next().await {
            let (link, session) = (sender.clone(), session.clone());
            wasm_bindgen_futures::spawn_local(async move {
                // A copy outside the page's shared memory (a stream refuses shared bytes)
                let copy = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
                copy.copy_from(&bytes);
                let answer = match call(&session, "request", &[copy.into()]).await {
                    Ok(reply) => wire::Response::decode(js_sys::Uint8Array::new(&reply).to_vec().as_slice()).map_err(|e| format!("the server's answer could not be read: {e}")),
                    Err(e) => Err(e.to_string()),
                };
                link.deliver(id, answer);
            });
        }
        sender.closed();
    });
    Ok((link, token))
}

async fn over_websocket(url: &str, host: String, how: Hello) -> StreamResult<(Arc<PageLink>, Option<String>)> {
    let socket = web_sys::WebSocket::new(url).map_err(js)?;
    socket.set_binary_type(web_sys::BinaryType::Arraybuffer);
    let (texts_tx, texts) = futures_channel::mpsc::unbounded::<Option<String>>();
    let (opened_tx, mut opened) = futures_channel::mpsc::unbounded::<bool>();
    let ready: Arc<OnceLock<Arc<PageLink>>> = Arc::default();

    let (tx, to) = (texts_tx.clone(), ready.clone());
    let on_message = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        if let Some(text) = data.as_string() {
            let _ = tx.unbounded_send(Some(text));
        } else if let Ok(buffer) = data.dyn_into::<js_sys::ArrayBuffer>() {
            let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
            let (Some(link), true) = (to.get(), bytes.len() >= 4) else { return };
            let id = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            link.deliver(id, wire::Response::decode(&bytes[4..]).map_err(|e| format!("the server's answer could not be read: {e}")));
        }
    });
    socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
    let tx = opened_tx.clone();
    let on_open = Closure::<dyn FnMut()>::new(move || {
        let _ = tx.unbounded_send(true);
    });
    socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();
    let (tx, opened_tx, to) = (texts_tx, opened_tx, ready.clone());
    let on_close = Closure::<dyn FnMut()>::new(move || {
        let _ = tx.unbounded_send(None);
        let _ = opened_tx.unbounded_send(false);
        if let Some(link) = to.get() {
            link.closed();
        }
    });
    socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    if opened.next().await != Some(true) {
        return Err(StreamError::protocol(format!("could not reach {url}")));
    }
    let mut control = Control::WebSocket { socket: socket.clone(), texts };
    let (welcome, token) = hello(&mut control, how).await?;
    let (out, mut requests) = futures_channel::mpsc::unbounded::<(u32, Vec<u8>)>();
    let link = Arc::new(PageLink { out, pending: Mutex::default(), next: AtomicU32::new(1), welcome, transport: Transport::WebSocket, host });
    let _ = ready.set(link.clone());
    // The one sender: frames from any thread leave from here, copied out of shared memory
    wasm_bindgen_futures::spawn_local(async move {
        while let Some((id, bytes)) = requests.next().await {
            let copy = js_sys::Uint8Array::new_with_length(4 + bytes.len() as u32);
            copy.subarray(0, 4).copy_from(&id.to_be_bytes());
            copy.subarray(4, 4 + bytes.len() as u32).copy_from(&bytes);
            if socket.send_with_array_buffer(&copy.buffer()).is_err() {
                break;
            }
        }
    });
    Ok((link, token))
}

/// What the page's server says before the page connects (`/lucem-serve.json` next to the page),
/// or `None` when no `lucem serve` served the page (a plain file server, GitHub Pages).
pub async fn serve_info() -> StreamResult<Option<ServeInfo>> {
    let window: JsValue = web_sys::window().ok_or_else(|| StreamError::protocol("no window"))?.into();
    let url = format!(".{}", crate::pair::SERVE_PATH);
    let Ok(response) = call(&window, "fetch", &[url.into()]).await else { return Ok(None) };
    if !js_sys::Reflect::get(&response, &"ok".into()).map_err(js)?.as_bool().unwrap_or(false) {
        return Ok(None);
    }
    let text = call(&response, "text", &[]).await?.as_string().unwrap_or_default();
    let info: ServeInfo = serde_json::from_str(&text).map_err(|e| StreamError::protocol(e.to_string()))?;
    Ok(info.server.is_some().then_some(info))
}
