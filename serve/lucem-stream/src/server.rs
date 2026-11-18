//! The server: accepts enrolled clients and answers their requests.
//!
//! What clients send is untrusted. Paths must resolve inside the data folder (no `..`, nothing
//! hidden, no link out of it) and name an image (the authority folder and key files are never
//! reachable); levels, channels and tiles are checked by the reader; every connection is limited
//! in its concurrent requests (see [`crate::tls`]) and the server in its connections.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use lucem_io::Pyramid;

use crate::auth::{Authority, Peer};
use crate::error::{StreamError, StreamResult};
use crate::protocol::{read_message, write_message, ImageInfo, Request, Response};

/// Closing code for a client whose certificate is not enrolled or is revoked.
pub const REFUSED_CLIENT: u32 = 1;
/// Clients at once by default.
pub const DEFAULT_MAX_CONNECTIONS: usize = 32;
/// Images kept open.
const OPEN_IMAGES: usize = 64;
/// How deep the listing looks under the data folder.
const LIST_DEPTH: usize = 4;

/// What a server serves.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// Clients open images under this folder.
    pub data: PathBuf,
}

/// A QUIC endpoint accepting clients.
pub struct Server {
    endpoint: quinn::Endpoint,
    max_connections: usize,
    authority: Option<Arc<Authority>>,
    /// Clients without a certificate may read (else they may only pair).
    anonymous: bool,
    /// The names in the server's certificate, given to a computer that pairs.
    names: Vec<String>,
    /// Its plugins ([`crate::plugin`]).
    plugins: Vec<Arc<dyn crate::plugin::ServerPlugin>>,
}

impl Server {
    /// Listens on `addr` (UDP).
    pub fn bind(addr: SocketAddr, config: quinn::ServerConfig) -> StreamResult<Self> {
        Ok(Self { endpoint: quinn::Endpoint::server(config, addr)?, max_connections: DEFAULT_MAX_CONNECTIONS, authority: None, anonymous: false, names: Vec::new(), plugins: Vec::new() })
    }

    /// Checks every client's certificate against `authority`'s registry after the handshake.
    pub fn with_authority(mut self, authority: Arc<Authority>) -> Self {
        self.authority = Some(authority);
        self
    }

    /// Lets clients without a certificate read (a trusted network only). Without it they may
    /// only pair.
    pub fn with_anonymous(mut self, anonymous: bool) -> Self {
        self.anonymous = anonymous;
        self
    }

    /// The names the server's certificate carries (a paired computer connects by one of them).
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.names = names;
        self
    }

    /// At most `n` clients at once (at least one).
    /// Answers calls to `plugins` (domain services: histology, models), by their ids.
    pub fn with_plugins(mut self, plugins: Vec<Arc<dyn crate::plugin::ServerPlugin>>) -> Self {
        self.plugins = plugins;
        self
    }

    pub fn with_max_connections(mut self, n: usize) -> Self {
        self.max_connections = n.max(1);
        self
    }

    pub fn local_addr(&self) -> StreamResult<SocketAddr> {
        Ok(self.endpoint.local_addr()?)
    }

    /// The next client whose handshake succeeds and who is allowed in. A client must first prove
    /// it owns its address (QUIC Retry), and clients beyond the limit are refused.
    pub async fn accept(&self) -> Option<(quinn::Connection, Peer)> {
        loop {
            let incoming = self.endpoint.accept().await?;
            if !incoming.remote_address_validated() && incoming.may_retry() {
                if let Err(e) = incoming.retry() {
                    tracing::debug!("QUIC retry failed: {e}");
                }
                continue;
            }
            if self.endpoint.open_connections() >= self.max_connections {
                tracing::warn!("refused {}: {} clients connected (the limit)", incoming.remote_address(), self.max_connections);
                incoming.refuse();
                continue;
            }
            let connection = match incoming.await {
                Ok(connection) => connection,
                Err(e) => {
                    tracing::warn!("QUIC handshake failed: {e}");
                    continue;
                }
            };
            match self.peer(&connection) {
                Some(peer) => return Some((connection, peer)),
                None => {
                    tracing::warn!("refused {}: its certificate is not enrolled or is revoked", connection.remote_address());
                    connection.close(quinn::VarInt::from_u32(REFUSED_CLIENT), b"client not enrolled or revoked");
                }
            }
        }
    }

    fn peer(&self, connection: &quinn::Connection) -> Option<Peer> {
        let Some(authority) = &self.authority else { return Some(Peer::anonymous()) };
        let certificates = connection.peer_identity().and_then(|id| id.downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>().ok());
        match certificates.as_deref().and_then(|c| c.first()) {
            Some(certificate) => authority.authorize(certificate),
            None if self.anonymous => Some(Peer::anonymous()),
            None => Some(Peer::pairing()),
        }
    }

    /// Stops accepting and closes every connection.
    pub fn close(&self) {
        self.endpoint.close(quinn::VarInt::from_u32(0), b"server closed");
    }

    /// Serves until the endpoint is closed.
    pub async fn run(self, options: ServeOptions) -> StreamResult<()> {
        let root = options.data.canonicalize()?;
        let images = Arc::new(Images { root, open: Mutex::new((HashMap::new(), VecDeque::new())) });
        let names = Arc::new(self.names.clone());
        let plugins = Arc::new(self.plugins.clone());
        while let Some((connection, peer)) = self.accept().await {
            tracing::info!("client {:?} connected from {}", peer.name, connection.remote_address());
            let images = images.clone();
            let (authority, names, peer, plugins) = (self.authority.clone(), names.clone(), Arc::new(peer), plugins.clone());
            tokio::spawn(async move {
                loop {
                    let Ok((send, recv)) = connection.accept_bi().await else { break };
                    let (images, authority, names, peer, plugins) = (images.clone(), authority.clone(), names.clone(), peer.clone(), plugins.clone());
                    tokio::spawn(async move {
                        if let Err(e) = answer(send, recv, images, authority.as_deref(), &names, &peer, plugins).await {
                            tracing::debug!("request failed: {e}");
                        }
                    });
                }
            });
        }
        Ok(())
    }
}

type OpenImages = (HashMap<PathBuf, Arc<Pyramid>>, VecDeque<PathBuf>);

struct Images {
    root: PathBuf,
    open: Mutex<OpenImages>,
}

/// Whether the path names an image the server may open.
fn is_image_name(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    name.ends_with(".zarr") || name.ends_with(".tif") || name.ends_with(".tiff")
}

impl Images {
    /// `relative`, checked: inside the data folder, nothing hidden, an image.
    fn resolve(&self, relative: &str) -> Result<PathBuf, String> {
        let path = Path::new(relative);
        let plain = path.components().all(|c| matches!(c, Component::Normal(n) if !n.to_string_lossy().starts_with('.')));
        if relative.is_empty() || !plain {
            return Err("that path is not allowed".into());
        }
        if !is_image_name(path) {
            return Err("that is not an image".into());
        }
        let full = self.root.join(path).canonicalize().map_err(|_| "no such image".to_string())?;
        if !full.starts_with(&self.root) {
            return Err("that path is not allowed".into());
        }
        Ok(full)
    }

    fn open(&self, relative: &str) -> Result<Arc<Pyramid>, String> {
        let full = self.resolve(relative)?;
        if let Some(p) = self.open.lock().map_err(|e| e.to_string())?.0.get(&full) {
            return Ok(p.clone());
        }
        let pyramid = Arc::new(Pyramid::open(&full).map_err(|e| e.to_string())?);
        let mut open = self.open.lock().map_err(|e| e.to_string())?;
        if open.0.insert(full.clone(), pyramid.clone()).is_none() {
            open.1.push_back(full);
        }
        while open.1.len() > OPEN_IMAGES {
            if let Some(old) = open.1.pop_front() {
                open.0.remove(&old);
            }
        }
        Ok(pyramid)
    }

    fn list(&self) -> Vec<String> {
        let mut out = Vec::new();
        walk(&self.root, &self.root, 0, &mut out);
        out.sort();
        out
    }
}

fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let path = e.path();
        if e.file_name().to_string_lossy().starts_with('.') || e.file_type().map_or(true, |t| t.is_symlink()) {
            continue;
        }
        if is_image_name(&path) {
            if let Ok(r) = path.strip_prefix(root) {
                out.push(r.to_string_lossy().into_owned());
            }
        } else if path.is_dir() && depth < LIST_DEPTH {
            walk(root, &path, depth + 1, out);
        }
    }
}

fn describe(p: &Pyramid) -> ImageInfo {
    ImageInfo {
        name: p.name.clone(),
        dtype: p.dtype.clone(),
        channels: p.channels.iter().map(|c| c.label.clone()).collect(),
        pixel_size_um: p.pixel_size_um,
        z_spacing_um: p.z_spacing_um,
        levels: p.levels.iter().map(|l| [l.depth, l.height, l.width, l.tile_d, l.tile_h, l.tile_w]).collect(),
        display: p.channels.iter().map(|c| crate::protocol::ChannelDisplay { color: c.color, window: c.window, range: c.range, active: c.active }).collect(),
    }
}

async fn answer(
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
    images: Arc<Images>,
    authority: Option<&Authority>,
    names: &[String],
    peer: &Peer,
    plugins: Arc<Vec<Arc<dyn crate::plugin::ServerPlugin>>>,
) -> StreamResult<()> {
    let request: Request = read_message(&mut recv).await?;
    let request = match (request, authority) {
        (Request::Pair { hello }, Some(authority)) => {
            crate::pairing::serve_pairing(authority, names, hello, &mut send, &mut recv).await?;
            send.finish()?;
            return Ok(());
        }
        (Request::Pair { .. }, None) => {
            write_message(&mut send, &Response::Error { reason: "this server does not pair computers".into() }).await?;
            send.finish()?;
            return Ok(());
        }
        (_, _) if peer.pairing_only => {
            write_message(&mut send, &Response::Error { reason: "this computer is not paired with the server: pair it first".into() }).await?;
            send.finish()?;
            return Ok(());
        }
        (request, _) => request,
    };
    let context = crate::plugin::PluginContext { root: images.root.clone(), peer: peer.name.clone(), may_write: peer.permission == crate::auth::Permission::Control };
    let result = tokio::task::spawn_blocking(move || work(&images, request, &plugins, &context)).await.map_err(StreamError::protocol)?;
    match result {
        Ok((response, payload)) => {
            write_message(&mut send, &response).await?;
            if let Some(values) = payload {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
                send.write_all(&bytes).await?;
            }
        }
        Err(reason) => write_message(&mut send, &Response::Error { reason }).await?,
    }
    send.finish()?;
    Ok(())
}

fn work(images: &Images, request: Request, plugins: &[Arc<dyn crate::plugin::ServerPlugin>], context: &crate::plugin::PluginContext) -> Result<(Response, Option<Vec<f32>>), String> {
    Ok(match request {
        Request::Plugin { plugin, method, body } => {
            let p = plugins.iter().find(|p| p.id() == plugin).ok_or_else(|| format!("this server has no plugin {plugin:?}"))?;
            (Response::Plugin { body: p.call(&method, body, context)? }, None)
        }
        Request::Pair { .. } => return Err("pairing is answered before this".into()),
        Request::List => (Response::Images { paths: images.list() }, None),
        Request::Info { path } => (Response::Info { info: describe(images.open(&path)?.as_ref()) }, None),
        Request::Tile { path, level, channel, z, tx, ty } => {
            let p = images.open(&path)?;
            if channel >= p.channels.len() {
                return Err(format!("no channel {channel}"));
            }
            let t = p.read_tile_at(level, channel, z, tx, ty).map_err(|e| e.to_string())?;
            (Response::Tile { width: t.width, height: t.height }, Some(t.data))
        }
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::auth::Permission;
    use crate::client::Session;
    use crate::tls::{client_config_with, server_config_with_clients, ServerIdentity, ServerTrust};

    fn images(root: &Path) -> Images {
        Images { root: root.canonicalize().unwrap(), open: Mutex::new((HashMap::new(), VecDeque::new())) }
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lucem-stream-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn paths_stay_inside_the_data_folder_and_name_images() {
        let d = temp("paths");
        fs::write(d.join("a.tif"), b"x").unwrap();
        fs::write(d.join("key.pem"), b"secret").unwrap();
        let i = images(&d);
        assert!(i.resolve("a.tif").is_ok());
        for bad in ["../a.tif", "/etc/passwd", ".auth/ca.key.pem", "key.pem", ""] {
            assert!(i.resolve(bad).is_err(), "{bad}");
        }
        assert_eq!(i.list(), vec!["a.tif".to_string()]);
        fs::remove_dir_all(&d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_data_folder_is_refused() {
        let d = temp("link");
        let outside = temp("outside");
        fs::write(outside.join("b.tif"), b"x").unwrap();
        std::os::unix::fs::symlink(outside.join("b.tif"), d.join("b.tif")).unwrap();
        assert!(images(&d).resolve("b.tif").is_err());
        assert!(images(&d).list().is_empty());
        fs::remove_dir_all(&d).unwrap();
        fs::remove_dir_all(&outside).unwrap();
    }

    /// An enrolled client lists and reads; a revoked one is turned away.
    #[tokio::test]
    async fn only_enrolled_clients_read_tiles() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let data = temp("data");
        let auth_dir = temp("auth");
        let image = data.join("a.ome.zarr");
        let options = lucem_io::WriterOptions { name: "a".into(), channels: vec!["c".into()], shape: [1, 8, 8], chunks: [1, 8, 8], pixel_size_um: None, z_spacing_um: None };
        let w = lucem_io::ZarrWriter::create::<f32>(&image, options).unwrap();
        w.write(0, [0, 0, 0], [1, 8, 8], &[1.5f32; 64]).unwrap();
        w.finish::<f32>().unwrap();

        let authority = Arc::new(Authority::open_or_create(&auth_dir.join("a")).unwrap());
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec!["localhost".into()] }, false).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority.clone());
        let addr = server.local_addr().unwrap();
        tokio::spawn(server.run(ServeOptions { data: data.clone() }));

        let file = auth_dir.join("alice.pem");
        authority.enroll("alice", Permission::Read).unwrap().write(&file).unwrap();
        let identity = crate::auth::ClientIdentity::from_file(&file).unwrap();
        let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity)).unwrap();
        let session = Session::connect(addr, "localhost", config.clone()).await.unwrap();
        assert_eq!(session.list().await.unwrap(), vec!["a.ome.zarr".to_string()]);
        assert_eq!(session.info("a.ome.zarr").await.unwrap().channels, vec!["c".to_string()]);
        let tile = session.tile("a.ome.zarr", 0, 0, 0, 0, 0).await.unwrap();
        assert_eq!((tile.width, tile.height), (8, 8));
        assert!(tile.data.iter().all(|v| *v == 1.5));
        assert!(matches!(session.tile("../x.tif", 0, 0, 0, 0, 0).await, Err(StreamError::Server(_))));
        session.close();

        authority.revoke("alice").unwrap();
        let denied = async {
            let s = Session::connect(addr, "localhost", config).await?;
            s.list().await
        };
        assert!(denied.await.is_err(), "a revoked client is refused");
        fs::remove_dir_all(&data).unwrap();
        fs::remove_dir_all(&auth_dir).unwrap();
    }

    /// A served image opened through [`crate::Remote`] is a pyramid like the file: same levels,
    /// channels and display, same values; its tiles come over the connection.
    #[test]
    fn a_served_image_reads_as_a_pyramid() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let data = temp("remote-data");
        let auth_dir = temp("remote-auth");
        let image = data.join("b.ome.zarr");
        let options = lucem_io::WriterOptions { name: "b".into(), channels: vec!["dapi".into(), "cd3".into()], shape: [2, 20, 12], chunks: [1, 8, 8], pixel_size_um: Some(0.5), z_spacing_um: Some(2.0) };
        let w = lucem_io::ZarrWriter::create::<f32>(&image, options).unwrap();
        // Whole chunks, each value its index in the volume (negated in the second channel)
        for z in 0..2 {
            for y0 in (0..20).step_by(8) {
                for x0 in (0..12).step_by(8) {
                    let (h, wd) = ((20 - y0).min(8), (12 - x0).min(8));
                    let values: Vec<f32> = (0..h).flat_map(|y| (0..wd).map(move |x| ((z * 20 + y0 + y) * 12 + x0 + x) as f32)).collect();
                    w.write(0, [z, y0, x0], [1, h, wd], &values).unwrap();
                    w.write(1, [z, y0, x0], [1, h, wd], &values.iter().map(|v| -v).collect::<Vec<_>>()).unwrap();
                }
            }
        }
        w.finish::<f32>().unwrap();

        let server_runtime = tokio::runtime::Runtime::new().unwrap();
        let authority = Arc::new(Authority::open_or_create(&auth_dir.join("a")).unwrap());
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec!["localhost".into()] }, false).unwrap();
        let addr = server_runtime.block_on(async {
            let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority.clone());
            let addr = server.local_addr().unwrap();
            tokio::spawn(server.run(ServeOptions { data: data.clone() }));
            addr
        });
        let file = auth_dir.join("carol.pem");
        authority.enroll("carol", Permission::Read).unwrap().write(&file).unwrap();
        let paired = crate::PairedServer { addr: addr.to_string(), names: vec!["localhost".into()], identity: file };
        let remote = crate::Remote::connect(&paired).unwrap();
        assert_eq!(remote.list().unwrap(), vec!["b.ome.zarr".to_string()]);
        let served = remote.open("b.ome.zarr").unwrap();
        let local = Pyramid::open(&image).unwrap();
        assert!(served.is_remote());
        assert_eq!(served.path.as_deref(), Some(Path::new(&format!("lucem://{addr}/b.ome.zarr"))));
        assert_eq!((served.levels.len(), served.depth(), served.width(), served.height()), (local.levels.len(), local.depth(), local.width(), local.height()));
        assert_eq!(served.channels, local.channels);
        assert_eq!((served.pixel_size_um, served.z_spacing_um), (Some(0.5), Some(2.0)));
        for level in 0..local.levels.len() {
            for channel in 0..2 {
                let l = &local.levels[level];
                let (a, b) = (local.read_region_at(level, channel, 1, 0, 0, l.width, l.height).unwrap(), served.read_region_at(level, channel, 1, 0, 0, l.width, l.height).unwrap());
                assert_eq!(a, b, "level {level} channel {channel}");
            }
        }
        // Read from many threads at once (the reader's pool)
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| served.read_block(0, 1, [0, 0, 0], [2, 20, 12]).unwrap());
            }
        });
        remote.close();
        drop(server_runtime);
        fs::remove_dir_all(&data).unwrap();
        fs::remove_dir_all(&auth_dir).unwrap();
    }

    /// A plugin answers its calls with what it may use of the server (who calls, whether it may
    /// write, paths under the data folder only); an unknown plugin is refused.
    #[tokio::test]
    async fn plugins_answer_their_calls() {
        struct Echo;
        impl crate::plugin::ServerPlugin for Echo {
            fn id(&self) -> &'static str {
                "echo"
            }
            fn call(&self, method: &str, body: serde_json::Value, context: &crate::plugin::PluginContext) -> Result<serde_json::Value, String> {
                if method == "path" {
                    let rel = body.as_str().unwrap_or_default();
                    return context.resolve(rel).map(|p| serde_json::json!(p.starts_with(&context.root)));
                }
                Ok(serde_json::json!({"method": method, "body": body, "peer": context.peer, "may_write": context.may_write}))
            }
        }
        let _ = rustls::crypto::ring::default_provider().install_default();
        let data = temp("plugin-data");
        let auth_dir = temp("plugin-auth");
        let authority = Arc::new(Authority::open_or_create(&auth_dir.join("a")).unwrap());
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: vec!["localhost".into()] }, false).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority.clone()).with_plugins(vec![Arc::new(Echo)]);
        let addr = server.local_addr().unwrap();
        tokio::spawn(server.run(ServeOptions { data: data.clone() }));
        let file = auth_dir.join("bob.pem");
        authority.enroll("bob", Permission::Read).unwrap().write(&file).unwrap();
        let identity = crate::auth::ClientIdentity::from_file(&file).unwrap();
        let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity)).unwrap();
        let session = Session::connect(addr, "localhost", config).await.unwrap();
        let answer = session.plugin_call("echo", "hello", serde_json::json!({"x": 1})).await.unwrap();
        assert_eq!(answer, serde_json::json!({"method": "hello", "body": {"x": 1}, "peer": "bob", "may_write": false}));
        assert_eq!(session.plugin_call("echo", "path", serde_json::json!("a/b.tif")).await.unwrap(), serde_json::json!(true));
        assert!(session.plugin_call("echo", "path", serde_json::json!("../x")).await.is_err(), "outside the data folder");
        assert!(session.plugin_call("nothing", "x", serde_json::Value::Null).await.is_err(), "no such plugin");
        session.close();
        fs::remove_dir_all(&data).unwrap();
        fs::remove_dir_all(&auth_dir).unwrap();
    }

    /// A computer pairs with a code, then reads with mutual TLS; before pairing it may do nothing
    /// else, and a wrong code pairs nothing.
    #[tokio::test]
    async fn a_computer_pairs_with_a_code() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let data = temp("pair-data");
        let auth_dir = temp("pair-auth");
        let image = data.join("b.ome.zarr");
        let options = lucem_io::WriterOptions { name: "b".into(), channels: vec!["c".into()], shape: [1, 8, 8], chunks: [1, 8, 8], pixel_size_um: None, z_spacing_um: None };
        let w = lucem_io::ZarrWriter::create::<f32>(&image, options).unwrap();
        w.write(0, [0, 0, 0], [1, 8, 8], &[2.0f32; 64]).unwrap();
        w.finish::<f32>().unwrap();

        let authority = Arc::new(Authority::open_or_create(&auth_dir.join("a")).unwrap());
        let names = vec!["localhost".to_string()];
        let tls = server_config_with_clients(&authority, &ServerIdentity::SelfSigned { names: names.clone() }, false).unwrap();
        let server = Server::bind("127.0.0.1:0".parse().unwrap(), tls.config).unwrap().with_authority(authority.clone()).with_names(names);
        let addr = server.local_addr().unwrap();
        tokio::spawn(server.run(ServeOptions { data: data.clone() }));

        // Unpaired: connected, but nothing to read
        let anonymous = client_config_with(&ServerTrust::DangerAcceptAnyCertificate, None).unwrap();
        let s = Session::connect(addr, "localhost", anonymous).await.unwrap();
        assert!(matches!(s.list().await, Err(StreamError::Server(_))), "an unpaired computer reads nothing");
        s.close();

        // A wrong code pairs nothing
        let code = authority.new_pairing_code(Permission::Read).unwrap();
        let wrong = format!("{}-0000-0000", code.id);
        assert!(crate::pairing::pair(addr, &wrong, "mallory").await.is_err());
        assert!(authority.clients().unwrap().is_empty());

        // The right code: paired, and the code is spent
        let paired = crate::pairing::pair(addr, &code.display(), "laptop").await.unwrap();
        assert_eq!(paired.names, vec!["localhost".to_string()]);
        assert!(crate::pairing::pair(addr, &code.display(), "again").await.is_err(), "a code works once");
        let identity = crate::auth::ClientIdentity::from_pem(paired.identity_pem.as_bytes()).unwrap();
        let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity)).unwrap();
        let session = Session::connect(addr, "localhost", config).await.unwrap();
        assert_eq!(session.list().await.unwrap(), vec!["b.ome.zarr".to_string()]);
        session.close();
        assert_eq!(authority.clients().unwrap()[0].name, "laptop");
        fs::remove_dir_all(&data).unwrap();
        fs::remove_dir_all(&auth_dir).unwrap();
    }
}
