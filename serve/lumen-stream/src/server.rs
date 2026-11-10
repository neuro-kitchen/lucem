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

use lumen_io::Pyramid;

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
}

impl Server {
    /// Listens on `addr` (UDP).
    pub fn bind(addr: SocketAddr, config: quinn::ServerConfig) -> StreamResult<Self> {
        Ok(Self { endpoint: quinn::Endpoint::server(config, addr)?, max_connections: DEFAULT_MAX_CONNECTIONS, authority: None })
    }

    /// Checks every client's certificate against `authority`'s registry after the handshake.
    pub fn with_authority(mut self, authority: Arc<Authority>) -> Self {
        self.authority = Some(authority);
        self
    }

    /// At most `n` clients at once (at least one).
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
            None => Some(Peer::anonymous()),
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
        while let Some((connection, peer)) = self.accept().await {
            tracing::info!("client {:?} connected from {}", peer.name, connection.remote_address());
            let images = images.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((send, recv)) = connection.accept_bi().await else { break };
                    let images = images.clone();
                    tokio::spawn(async move {
                        if let Err(e) = answer(send, recv, images).await {
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
    }
}

async fn answer(mut send: quinn::SendStream, mut recv: quinn::RecvStream, images: Arc<Images>) -> StreamResult<()> {
    let request: Request = read_message(&mut recv).await?;
    let result = tokio::task::spawn_blocking(move || work(&images, request)).await.map_err(StreamError::protocol)?;
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

fn work(images: &Images, request: Request) -> Result<(Response, Option<Vec<f32>>), String> {
    Ok(match request {
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
        let d = std::env::temp_dir().join(format!("lumen-stream-{name}-{}", std::process::id()));
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
        let options = lumen_io::WriterOptions { name: "a".into(), channels: vec!["c".into()], shape: [1, 8, 8], chunks: [1, 8, 8], pixel_size_um: None, z_spacing_um: None };
        let w = lumen_io::ZarrWriter::create::<f32>(&image, options).unwrap();
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
}
