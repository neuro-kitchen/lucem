//! The server for browser pages (`lucem serve --web`), on one port number:
//!
//! - **HTTP(S) on TCP**: the page's files (from the web folder, Brotli copies and ETags as
//!   dsp-kitchen's), [`SERVE_PATH`] (what the page needs before it connects: the WebTransport
//!   port and its certificate's hash), and a WebSocket at [`WS_PATH`], the fallback where UDP
//!   does not get through.
//! - **WebTransport on UDP** ([`WT_PATH`], HTTP/3 over QUIC, quinn underneath as for computers):
//!   the page's own connection. Its certificate is self-signed, ECDSA, valid 14 days (a browser
//!   accepts a certificate it cannot check only so, by its SHA-256, `serverCertificateHashes`);
//!   it is made at every start, so a server running longer than that is restarted (the page
//!   reads the new hash on its next load).
//!
//! **Who the page is.** First, as JSON ([`crate::pair`]): a token from an earlier visit, or a
//! pairing with a one-time code, which ends in a token (a page cannot hold a client
//! certificate). On WebTransport this is the first bidirectional stream (4-byte big-endian
//! lengths, then the JSON); on the WebSocket, text frames. Then the page's requests
//! ([`crate::wire`]): on WebTransport one stream per request (closing it cancels the request:
//! a superseded tile never ties the connection up), on the WebSocket binary frames with an id.
//! Both are answered by [`crate::engine::answer_wire`], as computers' requests are by
//! [`crate::engine::work`].

use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use futures_util::{SinkExt as _, StreamExt as _};
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use hyper::{Response, StatusCode};
use hyper_util::rt::TokioIo;
use prost::Message as _;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use crate::auth::{Authority, Peer};
use crate::engine::{answer_wire, Images, TileProcessor};
use crate::error::{StreamError, StreamResult};
pub use crate::pair::{ServeInfo, WebHello, WebWelcome, SERVE_PATH, WS_PATH, WT_PATH};
use crate::pair::{PairReply, PairToken};
use crate::plugin::{PluginContext, ServerPlugin};
use crate::wire;

/// Largest request a page may send (a request is small).
pub const MAX_WEB_REQUEST_BYTES: usize = 1 << 20;
/// Largest JSON message while a page says who it is.
const MAX_HELLO_BYTES: usize = 64 << 10;
/// How long a page has to say who it is.
const HELLO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// What the page may load and connect to: its own files and server only (no inline script: the
/// loader is `loader.js`; WebAssembly compiled from the page's own module; workers from the page
/// or the blobs the threads start from). `'unsafe-eval'`: `wasm_thread`, which starts the page's
/// worker threads, evaluates two constant strings (`eval("self")` and its script-path finder);
/// scripts still come from this origin only. To drop it: wasm_thread using `js_sys::global()` and
/// an explicit shim URL (an upstream patch). `connect-src` also names this computer's loopback
/// address: a page at `localhost` reaches WebTransport at 127.0.0.1 (a CSP source cannot name an
/// IPv6 literal with a wildcard port) (QUIC does not fall back from
/// ::1); elsewhere WebTransport is the page's own host, which `'self'` covers.
pub const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'; worker-src 'self' blob:; connect-src 'self' https://127.0.0.1:*; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// Whether a connection's `Origin` is the page's own (`authority`: the `host:port` it reached):
/// another site open in the same browser may not use the server, whatever it knows. No origin:
/// not a browser (a script, a test), which still needs a token or a code.
fn same_origin(origin: Option<&str>, authority: &str) -> bool {
    let Some(origin) = origin else { return true };
    let host = origin.strip_prefix("https://").or_else(|| origin.strip_prefix("http://")).unwrap_or("").trim_end_matches('/');
    if host.is_empty() {
        return false;
    }
    if host.eq_ignore_ascii_case(authority) {
        return true;
    }
    // This computer by any of its loopback names, on the same port: a page opened at localhost
    // reaches WebTransport at 127.0.0.1 (QUIC does not fall back from ::1 as TCP does)
    let split = |a: &str| a.rsplit_once(':').map(|(h, p)| (h.trim_matches(['[', ']']).to_ascii_lowercase(), p.to_string()));
    let loopback = |h: &str| h == "localhost" || h == "::1" || h.starts_with("127.");
    matches!((split(host), split(authority)), (Some((h1, p1)), Some((h2, p2))) if p1 == p2 && loopback(&h1) && loopback(&h2))
}

/// How the HTTP side is secured (WebTransport always has its own 14-day certificate).
pub enum WebTls {
    /// Plain HTTP, on a loopback address only: a browser treats `http://127.0.0.1` and
    /// `http://localhost` as secure (WebTransport and threads allowed), and nothing leaves the
    /// computer. A proxy holding a real certificate can stand in front.
    Plain,
    /// Plain HTTP on any address, asked for explicitly (a container's port published only to the
    /// host's loopback, or behind a proxy that holds the certificate).
    PlainOffLoopback,
    /// The server authority's own certificate: encrypted, but a browser warns until it is trusted.
    Authority,
    /// A certificate chain and key from files (a public authority's: no warning).
    Files(crate::tls::ServerIdentity),
}

/// What the browser endpoint serves.
pub struct WebOptions {
    /// TCP for HTTP(S) and the WebSocket, UDP (the same port) for WebTransport.
    pub bind: SocketAddr,
    /// The folder of the page's files; `None`: only the connections are served.
    pub web_root: Option<PathBuf>,
    pub tls: WebTls,
    /// The server's names (its certificates' and those it gives the pages it pairs).
    pub names: Vec<String>,
    /// How the server calls itself to pages.
    pub server_name: String,
    /// The compute runtime it processes on ("" when none).
    pub runtime: String,
    /// The app's samples (`lucem samples`), offered to pages as `samples/…`.
    pub samples: Option<PathBuf>,
    /// Tile answers kept for the pages (MiB): a tile asked twice (a pan back, a second page, the
    /// first screen of an image just opened) is sent from memory.
    pub tile_cache_mib: usize,
}

/// The default of [`WebOptions::tile_cache_mib`].
pub const DEFAULT_TILE_CACHE_MIB: usize = 512;
/// Most tiles read ahead when an image is opened (its coarsest levels, every channel).
const WARM_TILES: usize = 256;

/// A tile request's identity: the image, where, and the pipeline (its JSON) if processed.
type TileKey = (String, u32, u32, u32, u32, u32, Option<String>);

/// Encoded tile answers, least recently used dropped first.
struct TileCache {
    map: std::collections::HashMap<TileKey, (Arc<Vec<u8>>, u64)>,
    bytes: usize,
    budget: usize,
    clock: u64,
}

impl TileCache {
    fn new(budget: usize) -> Self {
        Self { map: Default::default(), bytes: 0, budget, clock: 0 }
    }

    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<u8>>> {
        self.clock += 1;
        let clock = self.clock;
        self.map.get_mut(key).map(|e| {
            e.1 = clock;
            e.0.clone()
        })
    }

    fn put(&mut self, key: TileKey, bytes: Arc<Vec<u8>>) {
        if bytes.len() > self.budget {
            return;
        }
        self.clock += 1;
        self.bytes += bytes.len();
        if let Some(old) = self.map.insert(key, (bytes, self.clock)) {
            self.bytes -= old.0.len();
        }
        while self.bytes > self.budget {
            let Some(oldest) = self.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| k.clone()) else { break };
            if let Some(e) = self.map.remove(&oldest) {
                self.bytes -= e.0.len();
            }
        }
    }
}

fn tile_key(t: &wire::GetTile) -> TileKey {
    (t.path.clone(), t.level, t.channel, t.z, t.tx, t.ty, t.processing.as_ref().map(|p| p.spec_json.clone()))
}

struct State {
    authority: Arc<Authority>,
    images: Arc<Images>,
    plugins: Arc<Vec<Arc<dyn ServerPlugin>>>,
    processor: Option<Arc<dyn TileProcessor>>,
    data_root: PathBuf,
    web_root: Option<PathBuf>,
    names: Vec<String>,
    server_name: String,
    runtime: String,
    serve_info: ServeInfo,
    tiles: std::sync::Mutex<TileCache>,
    /// Images already read ahead.
    warmed: std::sync::Mutex<std::collections::HashSet<String>>,
}

/// The browser endpoint, bound.
pub struct WebServer {
    listener: tokio::net::TcpListener,
    tls: Option<tokio_rustls::TlsAcceptor>,
    transport: wtransport::Endpoint<wtransport::endpoint::endpoint_side::Server>,
    state: Arc<State>,
}

impl WebServer {
    /// Binds `options.bind` (TCP and UDP), serving the images under `data` to pages the
    /// `authority` lets in, with these plugins and tile processor.
    pub async fn bind(options: WebOptions, authority: Arc<Authority>, data: &Path, plugins: Vec<Arc<dyn ServerPlugin>>, processor: Option<Arc<dyn TileProcessor>>) -> StreamResult<Self> {
        let WebOptions { bind, web_root, tls, names, server_name, runtime, samples, tile_cache_mib } = options;
        let chain = match &tls {
            WebTls::Plain if !bind.ip().is_loopback() => return Err(StreamError::Tls(format!("plain HTTP is only served on a loopback address, not on {bind} (--web-plain-http to serve it anyway)"))),
            WebTls::Plain | WebTls::PlainOffLoopback => None,
            WebTls::Authority => Some(authority.server_certificate(&names)?),
            WebTls::Files(identity) => Some(crate::tls::identity_chain(identity)?),
        };
        let tls = match chain {
            Some((chain, key)) => {
                let mut crypto = rustls::ServerConfig::builder().with_no_client_auth().with_single_cert(chain, key).map_err(|e| StreamError::Tls(e.to_string()))?;
                crypto.alpn_protocols = vec![b"http/1.1".to_vec()];
                Some(tokio_rustls::TlsAcceptor::from(Arc::new(crypto)))
            }
            None => None,
        };
        let listener = tokio::net::TcpListener::bind(bind).await?;
        let port = listener.local_addr()?.port();
        // WebTransport: the same port number, UDP; a certificate the browser checks by its hash
        let mut sans: Vec<String> = names.clone();
        for loopback in ["localhost", "127.0.0.1", "::1"] {
            if !sans.iter().any(|n| n == loopback) {
                sans.push(loopback.into());
            }
        }
        if !bind.ip().is_unspecified() && !sans.contains(&bind.ip().to_string()) {
            sans.push(bind.ip().to_string());
        }
        let identity = wtransport::Identity::self_signed(&sans).map_err(|e| StreamError::Tls(format!("the WebTransport certificate: {e}")))?;
        let certificate_sha256 = identity.certificate_chain().as_slice().first().map(|c| c.hash().as_ref().to_vec()).unwrap_or_default();
        let config = wtransport::ServerConfig::builder().with_bind_address(SocketAddr::new(bind.ip(), port)).with_identity(identity).keep_alive_interval(Some(std::time::Duration::from_secs(5))).build();
        let transport = wtransport::Endpoint::server(config)?;
        let root = data.canonicalize()?;
        let serve_info = ServeInfo { server: Some(server_name.clone()), webtransport_port: Some(port), certificate_sha256 };
        let tiles = std::sync::Mutex::new(TileCache::new(tile_cache_mib << 20));
        let data_root = root.clone();
        let state = State { authority, images: Arc::new(Images::new(root).with_samples(samples)), plugins: Arc::new(plugins), processor, data_root, web_root, names, server_name, runtime, serve_info, tiles, warmed: Default::default() };
        Ok(Self { listener, tls, transport, state: Arc::new(state) })
    }

    /// The TCP address (HTTP); WebTransport is on the same port, UDP.
    pub fn local_addr(&self) -> StreamResult<SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Whether pages are served over HTTPS (else plain HTTP).
    pub fn is_tls(&self) -> bool {
        self.tls.is_some()
    }

    /// What a page is told before it connects.
    pub fn serve_info(&self) -> &ServeInfo {
        &self.state.serve_info
    }

    /// Serves pages until the task is dropped.
    pub async fn run(self) {
        let Self { listener, tls, transport, state } = self;
        let http = {
            let state = state.clone();
            async move {
                loop {
                    let (tcp, remote) = match listener.accept().await {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!("web: accept failed: {e}");
                            continue;
                        }
                    };
                    let (tls, state) = (tls.clone(), state.clone());
                    tokio::spawn(async move {
                        let service = hyper::service::service_fn(move |request| handle(request, state.clone()));
                        let http = hyper::server::conn::http1::Builder::new();
                        let served = match tls {
                            Some(tls) => match tls.accept(tcp).await {
                                Ok(stream) => http.serve_connection(TokioIo::new(stream), service).with_upgrades().await,
                                Err(e) => {
                                    tracing::debug!("web: TLS handshake with {remote} failed: {e}");
                                    return;
                                }
                            },
                            None => http.serve_connection(TokioIo::new(tcp), service).with_upgrades().await,
                        };
                        if let Err(e) = served {
                            tracing::debug!("web: connection with {remote} ended: {e}");
                        }
                    });
                }
            }
        };
        let webtransport = async move {
            loop {
                let incoming = transport.accept().await;
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = webtransport_session(incoming, state).await {
                        tracing::debug!("webtransport: session ended: {e}");
                    }
                });
            }
        };
        tokio::join!(http, webtransport);
    }
}

// ---- who the page is (both transports) ---------------------------------------------------------

/// JSON messages while a page says who it is, whatever carries them.
trait Texts {
    fn send_text(&mut self, text: String) -> impl Future<Output = StreamResult<()>> + Send;
    fn next_text(&mut self) -> impl Future<Output = StreamResult<String>> + Send;
}

async fn send_json(texts: &mut impl Texts, message: &impl serde::Serialize) -> StreamResult<()> {
    texts.send_text(serde_json::to_string(message)?).await
}

async fn next_json<M: for<'de> serde::Deserialize<'de>>(texts: &mut impl Texts) -> StreamResult<M> {
    Ok(serde_json::from_str(&texts.next_text().await?)?)
}

/// Who the page is: by its token, or by pairing now (it then gets a token).
async fn identify(texts: &mut impl Texts, state: &State) -> StreamResult<Option<Peer>> {
    match next_json::<WebHello>(texts).await? {
        WebHello::Token { token } => Ok(state.authority.authorize_token(&token)),
        WebHello::Pair { pair } => {
            let pairing = match state.authority.pairing_start(&pair) {
                Ok((pairing, reply)) => {
                    send_json(texts, &reply).await?;
                    pairing
                }
                Err(refusal) => {
                    send_json(texts, &refusal).await?;
                    return Ok(None);
                }
            };
            let proof: PairToken = next_json(texts).await?;
            let reply = state.authority.pairing_finish_token(pairing, &proof, &state.names);
            let peer = match &reply {
                PairReply::Token { token, .. } => state.authority.authorize_token(token),
                _ => None,
            };
            send_json(texts, &reply).await?;
            Ok(peer)
        }
    }
}

/// [`identify`] within [`HELLO_TIMEOUT`], then the welcome; `None`: not let in.
async fn let_in(texts: &mut impl Texts, state: &State) -> StreamResult<Peer> {
    let peer = match tokio::time::timeout(HELLO_TIMEOUT, identify(texts, state)).await {
        Ok(Ok(Some(peer))) => peer,
        Ok(Ok(None)) => return Err(StreamError::protocol("not let in")),
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(StreamError::protocol("the page did not say who it is")),
    };
    let plugins = state.plugins.iter().map(|p| p.id().to_string()).collect();
    let welcome = WebWelcome { server_name: state.server_name.clone(), server_version: env!("CARGO_PKG_VERSION").into(), peer_name: peer.name.clone(), permission: peer.permission, runtime: state.runtime.clone(), plugins };
    send_json(texts, &welcome).await?;
    tracing::info!("web: {} connected ({})", peer.name, peer.permission);
    Ok(peer)
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Answers one request off the async threads (reading files, running a pipeline): the encoded
/// response. Tiles come from the cache when they are in it; an image's description starts
/// reading its coarsest levels ahead, so its first screen is in memory when the page asks.
async fn answer(state: Arc<State>, peer: Arc<Peer>, request: wire::Request) -> Arc<Vec<u8>> {
    let key = match &request.kind {
        Some(wire::request::Kind::Tile(t)) => Some(tile_key(t)),
        _ => None,
    };
    if let Some(hit) = key.as_ref().and_then(|k| lock(&state.tiles).get(k)) {
        return hit;
    }
    if let Some(wire::request::Kind::Info(i)) = &request.kind {
        warm(&state, &peer, &i.path);
    }
    let context = context_of(&state, &peer);
    let s = state.clone();
    let response = tokio::task::spawn_blocking(move || answer_wire(&s.images, &s.plugins, s.processor.as_deref(), request, &context)).await.unwrap_or_else(|e| wire::error(format!("the request failed: {e}")));
    let bytes = Arc::new(response.encode_to_vec());
    if let (Some(key), Some(wire::response::Kind::Tile(_))) = (key, &response.kind) {
        lock(&state.tiles).put(key, bytes.clone());
    }
    bytes
}

fn context_of(state: &State, peer: &Peer) -> PluginContext {
    PluginContext { root: state.images.root().to_path_buf(), peer: peer.name.clone(), may_write: peer.permission == crate::auth::Permission::Control, samples: state.images.samples().map(Path::to_path_buf) }
}

/// Reads the coarsest levels of `path` (every channel, the first plane) into the tile cache, in
/// the background, once per image: at most [`WARM_TILES`] tiles.
fn warm(state: &Arc<State>, peer: &Arc<Peer>, path: &str) {
    if !lock(&state.warmed).insert(path.to_string()) {
        return;
    }
    let (state, path, context) = (state.clone(), path.to_string(), context_of(state, peer));
    tokio::task::spawn_blocking(move || {
        let Ok(p) = state.images.open(&path) else { return };
        let mut asks = Vec::new();
        for level in (0..p.levels.len()).rev() {
            let l = &p.levels[level];
            let n = l.tiles_x() * l.tiles_y() * p.channels.len();
            if asks.len() + n > WARM_TILES {
                break;
            }
            for channel in 0..p.channels.len() {
                for ty in 0..l.tiles_y() {
                    for tx in 0..l.tiles_x() {
                        asks.push(wire::GetTile { path: path.clone(), level: level as u32, channel: channel as u32, z: 0, tx: tx as u32, ty: ty as u32, processing: None });
                    }
                }
            }
        }
        for ask in asks {
            let key = tile_key(&ask);
            if lock(&state.tiles).get(&key).is_some() {
                continue;
            }
            let response = answer_wire(&state.images, &state.plugins, None, wire::Request { kind: Some(wire::request::Kind::Tile(ask)) }, &context);
            if matches!(response.kind, Some(wire::response::Kind::Tile(_))) {
                lock(&state.tiles).put(key, Arc::new(response.encode_to_vec()));
            }
        }
    });
}

// ---- WebTransport ------------------------------------------------------------------------------

/// A WebTransport stream's messages: a 4-byte big-endian length, then the bytes.
async fn read_frame(recv: &mut wtransport::RecvStream, max: usize) -> StreamResult<Vec<u8>> {
    let mut len = [0u8; 4];
    recv.read_exact(&mut len).await.map_err(StreamError::protocol)?;
    let len = u32::from_be_bytes(len) as usize;
    if len > max {
        return Err(StreamError::TooLarge { len, max });
    }
    let mut body = vec![0u8; len];
    recv.read_exact(&mut body).await.map_err(StreamError::protocol)?;
    Ok(body)
}

async fn write_frame(send: &mut wtransport::SendStream, bytes: &[u8]) -> StreamResult<()> {
    let mut frame = Vec::with_capacity(4 + bytes.len());
    frame.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(bytes);
    send.write_all(&frame).await.map_err(StreamError::protocol)
}

/// The first stream of a session: who the page is.
struct Control {
    send: wtransport::SendStream,
    recv: wtransport::RecvStream,
}

impl Texts for Control {
    async fn send_text(&mut self, text: String) -> StreamResult<()> {
        write_frame(&mut self.send, text.as_bytes()).await
    }

    async fn next_text(&mut self) -> StreamResult<String> {
        String::from_utf8(read_frame(&mut self.recv, MAX_HELLO_BYTES).await?).map_err(StreamError::protocol)
    }
}

async fn webtransport_session(incoming: wtransport::endpoint::IncomingSession, state: Arc<State>) -> StreamResult<()> {
    let request = incoming.await.map_err(StreamError::protocol)?;
    if request.path() != WT_PATH {
        request.not_found().await;
        return Ok(());
    }
    if !same_origin(request.origin(), request.authority()) {
        tracing::warn!("webtransport: refused a session from {:?} (not the page's origin)", request.origin());
        request.forbidden().await;
        return Ok(());
    }
    let connection = request.accept().await.map_err(StreamError::protocol)?;
    let (send, recv) = connection.accept_bi().await.map_err(StreamError::protocol)?;
    let mut control = Control { send, recv };
    let peer = match let_in(&mut control, &state).await {
        Ok(peer) => Arc::new(peer),
        Err(e) => {
            connection.close(wtransport::VarInt::from_u32(1), b"not let in");
            return Err(e);
        }
    };
    // One stream per request, answered concurrently; a stream the page closes is a request it
    // no longer wants (the answer's write fails, nothing else waits on it)
    loop {
        let Ok((mut send, mut recv)) = connection.accept_bi().await else { break };
        let (state, peer) = (state.clone(), peer.clone());
        tokio::spawn(async move {
            let response = match read_frame(&mut recv, MAX_WEB_REQUEST_BYTES).await.and_then(|b| wire::Request::decode(b.as_slice()).map_err(StreamError::protocol)) {
                Ok(request) => answer(state, peer, request).await,
                Err(e) => Arc::new(wire::error(format!("not a request: {e}")).encode_to_vec()),
            };
            if write_frame(&mut send, &response).await.is_ok() {
                let _ = send.finish().await;
            }
        });
    }
    Ok(())
}

// ---- HTTP and the WebSocket --------------------------------------------------------------------

type Reply = Response<Full<Bytes>>;

fn reply(status: StatusCode, content_type: &str, body: impl Into<Bytes>) -> Reply {
    Response::builder()
        .status(status)
        .header("content-type", content_type)
        // The page computes in threads and shared memory only when it is isolated like this
        .header("cross-origin-opener-policy", "same-origin")
        .header("cross-origin-embedder-policy", "require-corp")
        .header("x-content-type-options", "nosniff")
        .header("content-security-policy", CONTENT_SECURITY_POLICY)
        .header("referrer-policy", "no-referrer")
        .body(Full::new(body.into()))
        .expect("a response")
}

async fn handle(mut request: hyper::Request<Incoming>, state: Arc<State>) -> Result<Reply, std::convert::Infallible> {
    if request.uri().path() == WS_PATH {
        let origin = request.headers().get("origin").and_then(|v| v.to_str().ok());
        let host = request.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or_default();
        if !same_origin(origin, host) {
            tracing::warn!("web: refused a WebSocket from {origin:?} (not the page's origin)");
            return Ok(reply(StatusCode::FORBIDDEN, "text/plain", "not this page's origin"));
        }
        return Ok(upgrade(&mut request, state));
    }
    // The page's loader asks this first: answered, the page is a client of this server (and
    // learns where WebTransport is); not found, it runs on its own
    if request.uri().path().ends_with(SERVE_PATH) {
        let info = serde_json::to_string(&state.serve_info).unwrap_or_else(|_| "{}".into());
        return Ok(reply(StatusCode::OK, "application/json", info));
    }
    if request.uri().path().starts_with("/api/") {
        let origin = request.headers().get("origin").and_then(|v| v.to_str().ok());
        let host = request.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or_default();
        if !same_origin(origin, host) {
            return Ok(reply(StatusCode::FORBIDDEN, "text/plain", "not this page's origin"));
        }
        return Ok(crate::web_api::handle(request, &state.data_root).await);
    }
    // HEAD: the same answer as GET without its body (link checkers, curl -I)
    let head = request.method() == hyper::Method::HEAD;
    if request.method() != hyper::Method::GET && !head {
        return Ok(reply(StatusCode::METHOD_NOT_ALLOWED, "text/plain", "GET and HEAD only"));
    }
    let Some(root) = state.web_root.clone() else {
        return Ok(reply(StatusCode::NOT_FOUND, "text/plain", "this server has no web app (lucem serve --web <folder>)"));
    };
    let path = request.uri().path().to_string();
    let header = |name: &str| request.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
    let (accepts_brotli, cached) = (header("accept-encoding").split(',').any(|e| e.trim().starts_with("br")), header("if-none-match"));
    let mut r = tokio::task::spawn_blocking(move || file(&root, &path, accepts_brotli, &cached)).await.unwrap_or_else(|_| reply(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "failed"));
    if head {
        *r.body_mut() = Full::new(Bytes::new());
    }
    Ok(r)
}

/// The file at `path` under `root`: plain components only, never out of the folder; its Brotli
/// copy (`<file>.br`) when the browser takes it; an ETag (size and time) so a page that has it
/// gets "not modified". (As dsp-kitchen's.)
fn file(root: &Path, path: &str, accepts_brotli: bool, cached: &str) -> Reply {
    let relative = path.trim_start_matches('/');
    let relative = if relative.is_empty() { "index.html" } else { relative };
    let plain = Path::new(relative).components().all(|c| matches!(c, Component::Normal(n) if !n.to_string_lossy().starts_with('.')));
    if !plain {
        return reply(StatusCode::NOT_FOUND, "text/plain", "not found");
    }
    let full = root.join(relative);
    let inside = match (full.canonicalize(), root.canonicalize()) {
        (Ok(f), Ok(r)) => f.starts_with(r) && f.is_file(),
        _ => false,
    };
    if !inside {
        return reply(StatusCode::NOT_FOUND, "text/plain", "not found");
    }
    let content_type = match full.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css; charset=utf-8",
        Some("json") => "application/json",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("ttf") => "font/ttf",
        _ => "application/octet-stream",
    };
    let compressed = accepts_brotli.then(|| full.with_file_name(format!("{}.br", full.file_name().unwrap_or_default().to_string_lossy()))).filter(|p| p.is_file());
    let sent = compressed.as_deref().unwrap_or(&full);
    let tag = std::fs::metadata(sent).ok().map(|m| {
        let time = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        format!("\"{:x}-{time:x}{}\"", m.len(), if compressed.is_some() { "-br" } else { "" })
    });
    let mut response = match (&tag, std::fs::read(sent)) {
        (Some(tag), _) if !cached.is_empty() && cached.split(',').any(|t| t.trim() == tag) => reply(StatusCode::NOT_MODIFIED, content_type, Bytes::new()),
        (_, Ok(bytes)) => reply(StatusCode::OK, content_type, bytes),
        (_, Err(_)) => return reply(StatusCode::NOT_FOUND, "text/plain", "not found"),
    };
    let headers = response.headers_mut();
    if let Some(tag) = tag.and_then(|t| hyper::header::HeaderValue::from_str(&t).ok()) {
        headers.insert(hyper::header::ETAG, tag);
    }
    headers.insert(hyper::header::CACHE_CONTROL, hyper::header::HeaderValue::from_static("no-cache"));
    headers.insert(hyper::header::VARY, hyper::header::HeaderValue::from_static("accept-encoding"));
    if compressed.is_some() {
        headers.insert(hyper::header::CONTENT_ENCODING, hyper::header::HeaderValue::from_static("br"));
    }
    response
}

/// Answers the WebSocket handshake and hands the connection to [`websocket_session`].
fn upgrade(request: &mut hyper::Request<Incoming>, state: Arc<State>) -> Reply {
    let header = |name: &str| request.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_ascii_lowercase);
    let wants = header("upgrade").is_some_and(|v| v == "websocket");
    let Some(key) = request.headers().get("sec-websocket-key").cloned().filter(|_| wants) else {
        return reply(StatusCode::BAD_REQUEST, "text/plain", "a WebSocket is expected here");
    };
    let accept = tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes());
    let upgraded = hyper::upgrade::on(request);
    tokio::spawn(async move {
        match upgraded.await {
            Ok(io) => {
                let config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default().max_message_size(Some(MAX_WEB_REQUEST_BYTES)).max_frame_size(Some(MAX_WEB_REQUEST_BYTES));
                let socket = WebSocketStream::from_raw_socket(TokioIo::new(io), Role::Server, Some(config)).await;
                if let Err(e) = websocket_session(socket, state).await {
                    tracing::debug!("web: session ended: {e}");
                }
            }
            Err(e) => tracing::debug!("web: upgrade failed: {e}"),
        }
    });
    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header("upgrade", "websocket")
        .header("connection", "Upgrade")
        .header("sec-websocket-accept", accept)
        .body(Full::new(Bytes::new()))
        .expect("a response")
}

fn web_error(e: impl std::fmt::Display) -> StreamError {
    StreamError::Protocol(e.to_string())
}

impl<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send> Texts for WebSocketStream<S> {
    async fn send_text(&mut self, text: String) -> StreamResult<()> {
        self.send(Message::text(text)).await.map_err(web_error)
    }

    async fn next_text(&mut self) -> StreamResult<String> {
        loop {
            match self.next().await {
                Some(Ok(Message::Text(text))) => return Ok(text.as_str().to_string()),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(_)) => return Err(StreamError::protocol("a text frame was expected")),
                Some(Err(e)) => return Err(web_error(e)),
                None => return Err(StreamError::protocol("the page left")),
            }
        }
    }
}

/// One page on the WebSocket: who it is, then its requests, each answered with its id.
async fn websocket_session<S>(mut socket: WebSocketStream<S>, state: Arc<State>) -> StreamResult<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let peer = match let_in(&mut socket, &state).await {
        Ok(peer) => Arc::new(peer),
        Err(e) => {
            let _ = socket.close(None).await;
            return Err(e);
        }
    };
    let (mut sink, mut stream) = socket.split();
    // Answers are computed concurrently and written by one task, in the order they finish
    let (answers, mut ready) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    let writer = tokio::spawn(async move {
        while let Some(frame) = ready.recv().await {
            if sink.send(Message::binary(frame)).await.is_err() {
                break;
            }
        }
    });
    while let Some(frame) = stream.next().await {
        let bytes = match frame.map_err(web_error)? {
            Message::Binary(bytes) => bytes,
            Message::Close(_) => break,
            _ => continue,
        };
        if bytes.len() < 4 {
            break;
        }
        let id: [u8; 4] = bytes[..4].try_into().expect("four bytes");
        let request = wire::Request::decode(&bytes[4..]);
        let (state, peer, answers) = (state.clone(), peer.clone(), answers.clone());
        tokio::spawn(async move {
            let response = match request {
                Ok(request) => answer(state, peer, request).await,
                Err(e) => Arc::new(wire::error(format!("not a request: {e}")).encode_to_vec()),
            };
            let mut frame = Vec::with_capacity(4 + response.len());
            frame.extend_from_slice(&id);
            frame.extend_from_slice(&response);
            let _ = answers.send(frame).await;
        });
    }
    drop(answers);
    writer.abort();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Permission;
    use crate::pair::ClientPairing;

    struct Pipe(tokio::sync::mpsc::UnboundedSender<String>, tokio::sync::mpsc::UnboundedReceiver<String>);

    impl Texts for Pipe {
        async fn send_text(&mut self, text: String) -> StreamResult<()> {
            self.0.send(text).map_err(StreamError::protocol)
        }
        async fn next_text(&mut self) -> StreamResult<String> {
            self.1.recv().await.ok_or_else(|| StreamError::protocol("closed"))
        }
    }

    fn state(dir: &Path) -> State {
        let authority = Arc::new(Authority::open_or_create(&dir.join("auth")).unwrap());
        std::fs::create_dir_all(dir.join("data")).unwrap();
        let data_root = dir.join("data").canonicalize().unwrap();
        let images = Arc::new(Images::new(data_root.clone()));
        State { authority, images, plugins: Arc::default(), processor: None, data_root, web_root: None, names: vec!["localhost".into()], server_name: "test".into(), runtime: String::new(), serve_info: ServeInfo { server: None, webtransport_port: None, certificate_sha256: Vec::new() }, tiles: std::sync::Mutex::new(TileCache::new(1 << 20)), warmed: Default::default() }
    }

    /// A page pairs with a code (and gets a token), then comes back with the token; a wrong
    /// token is not let in.
    #[tokio::test]
    async fn a_page_pairs_then_comes_back_with_its_token() {
        let dir = std::env::temp_dir().join(format!("lucem-web-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = Arc::new(state(&dir));
        let code = state.authority.new_pairing_code(Permission::Read).unwrap();

        let (to_server, server_in) = tokio::sync::mpsc::unbounded_channel();
        let (to_page, page_in) = tokio::sync::mpsc::unbounded_channel();
        let server = {
            let state = state.clone();
            tokio::spawn(async move { let_in(&mut Pipe(to_page, server_in), &state).await.map(|p| p.name) })
        };
        let mut page = Pipe(to_server, page_in);
        let (pairing, hello) = ClientPairing::start(&code.display(), "browser-1").unwrap();
        send_json(&mut page, &WebHello::Pair { pair: hello }).await.unwrap();
        let (pairing, proof) = pairing.token(next_json(&mut page).await.unwrap()).unwrap();
        send_json(&mut page, &proof).await.unwrap();
        let token = pairing.finish(next_json(&mut page).await.unwrap()).unwrap().token;
        let welcome: WebWelcome = next_json(&mut page).await.unwrap();
        assert_eq!((welcome.peer_name.as_str(), welcome.permission), ("browser-1", Permission::Read));
        assert_eq!(server.await.unwrap().unwrap(), "browser-1");

        assert_eq!(state.authority.authorize_token(&token).map(|p| p.name), Some("browser-1".into()));
        assert!(state.authority.authorize_token("0000").is_none(), "a wrong token");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn files_stay_inside_the_web_folder() {
        let dir = std::env::temp_dir().join(format!("lucem-web-files-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<p>").unwrap();
        assert_eq!(file(&dir, "/", false, "").status(), StatusCode::OK);
        assert_eq!(file(&dir, "/../etc/passwd", false, "").status(), StatusCode::NOT_FOUND);
        assert_eq!(file(&dir, "/.hidden", false, "").status(), StatusCode::NOT_FOUND);
        let r = file(&dir, "/index.html", false, "");
        assert_eq!(r.headers().get("cross-origin-embedder-policy").unwrap(), "require-corp");
        assert!(r.headers().get("content-security-policy").unwrap().to_str().unwrap().contains("script-src 'self'"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn only_the_pages_own_origin_connects() {
        assert!(same_origin(Some("http://127.0.0.1:8443"), "127.0.0.1:8443"));
        assert!(same_origin(Some("https://Lab.example:8443"), "lab.example:8443"));
        assert!(same_origin(None, "127.0.0.1:8443"), "not a browser");
        assert!(!same_origin(Some("https://evil.example"), "127.0.0.1:8443"));
        assert!(!same_origin(Some("http://127.0.0.1:9000"), "127.0.0.1:8443"), "another port is another origin");
        assert!(!same_origin(Some("null"), "127.0.0.1:8443"));
        // This computer by another loopback name, same port
        assert!(same_origin(Some("http://localhost:8443"), "127.0.0.1:8443"));
        assert!(same_origin(Some("http://localhost:8443"), "[::1]:8443"));
        assert!(!same_origin(Some("http://localhost:9000"), "127.0.0.1:8443"), "another port");
        assert!(!same_origin(Some("http://lab.example:8443"), "127.0.0.1:8443"), "not loopback");
    }

    #[test]
    fn the_tile_cache_keeps_the_newest_within_its_budget() {
        let key = |i: u32| ("a".to_string(), 0, 0, 0, i, 0, None);
        let mut c = TileCache::new(100);
        c.put(key(1), Arc::new(vec![0; 40]));
        c.put(key(2), Arc::new(vec![0; 40]));
        assert!(c.get(&key(1)).is_some(), "1 is used again: 2 is now the oldest");
        c.put(key(3), Arc::new(vec![0; 40]));
        assert!(c.get(&key(2)).is_none() && c.get(&key(1)).is_some() && c.get(&key(3)).is_some());
        assert_eq!(c.bytes, 80);
        c.put(key(4), Arc::new(vec![0; 200]));
        assert!(c.get(&key(4)).is_none(), "larger than the whole budget: not kept");
    }
}
