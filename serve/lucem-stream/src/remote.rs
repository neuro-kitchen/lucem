//! A paired server's images used as if they were here: [`Remote`] connects with this computer's
//! identity for that server and opens a served image as a [`Pyramid`] whose tiles come over the
//! connection ([`lucem_io::TileSource`]), so everything that reads a pyramid (the viewer, the
//! operators, the analysis) works on it unchanged.
//!
//! A served image is named `lucem://<host:port>/<path inside the server's data folder>`.
//!
//! The paired servers are the identities `lucem pair` writes in [`servers_dir`] (each a PEM,
//! readable by this user only), with a record beside each (`<file>.json`: the address and the names
//! its certificate carries).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lucem_io::{ChannelInfo, LevelInfo, Pyramid, TileSource};
use serde::{Deserialize, Serialize};

use crate::auth::ClientIdentity;
use crate::client::Session;
use crate::error::{StreamError, StreamResult};
use crate::protocol::ImageInfo;
use crate::tls::{client_config_with, ServerTrust};

/// How a served image's name starts.
pub const REMOTE_SCHEME: &str = "lucem://";

/// Where paired servers' identities live: `$XDG_CONFIG_HOME/lucem/servers` (else
/// `~/.config/lucem/servers`).
pub fn servers_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(std::env::temp_dir)
        .join("lucem")
        .join("servers")
}

/// A server this computer is paired with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedServer {
    /// `host:port`.
    pub addr: String,
    /// The names its certificate carries (the first is used to connect).
    pub names: Vec<String>,
    /// This computer's identity for it (PEM).
    #[serde(skip)]
    pub identity: PathBuf,
}

impl PairedServer {
    /// The record beside an identity file.
    pub fn record_path(identity: &Path) -> PathBuf {
        identity.with_extension("json")
    }

    /// Writes the record beside `self.identity`.
    pub fn save(&self) -> StreamResult<()> {
        std::fs::write(Self::record_path(&self.identity), serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }

    /// The servers paired in `dir` (sorted by address). An identity without its record (paired
    /// by an older lucem) is read from its file name (`host_port.pem`) and the name `localhost`.
    pub fn all_in(dir: &Path) -> Vec<Self> {
        let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
        let mut out: Vec<Self> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "pem"))
            .filter_map(|identity| {
                let record = std::fs::read(Self::record_path(&identity)).ok().and_then(|b| serde_json::from_slice::<Self>(&b).ok());
                let mut server = record.or_else(|| {
                    let stem = identity.file_stem()?.to_string_lossy().into_owned();
                    let (host, port) = stem.rsplit_once('_')?;
                    port.parse::<u16>().ok()?;
                    let host = if host.contains('_') { format!("[{}]", host.replace('_', ":")) } else { host.to_string() };
                    Some(Self { addr: format!("{host}:{port}"), names: vec!["localhost".into()], identity: PathBuf::new() })
                })?;
                server.identity = identity;
                Some(server)
            })
            .collect();
        out.sort_by(|a, b| a.addr.cmp(&b.addr));
        out
    }

    /// The servers paired on this computer.
    pub fn all() -> Vec<Self> {
        Self::all_in(&servers_dir())
    }
}

/// Splits `lucem://<addr>/<path>` into the address and the path.
pub fn split_remote(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(REMOTE_SCHEME)?;
    let (addr, path) = rest.split_once('/')?;
    (!addr.is_empty() && !path.is_empty()).then_some((addr, path))
}

/// A served image's name.
pub fn remote_name(addr: &str, path: &str) -> String {
    format!("{REMOTE_SCHEME}{addr}/{path}")
}

/// A connection to a paired server, usable from any thread that is not running async tasks of
/// its own (the app's background threads, the reader's thread pool).
pub struct Remote {
    // Dropped before the runtime it runs on
    session: Session,
    runtime: tokio::runtime::Runtime,
    addr: String,
}

impl std::fmt::Debug for Remote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Remote").field("addr", &self.addr).finish_non_exhaustive()
    }
}

impl Remote {
    /// Connects to `server` with this computer's identity for it.
    pub fn connect(server: &PairedServer) -> StreamResult<Arc<Self>> {
        // The process's TLS provider (an app has none set; the CLI sets the same)
        let _ = rustls::crypto::ring::default_provider().install_default();
        let identity = ClientIdentity::from_file(&server.identity)?;
        let config = client_config_with(&ServerTrust::from_identity(&identity), Some(&identity))?;
        let name = server.names.first().map_or("localhost", String::as_str);
        Self::connect_with(&server.addr, name, config)
    }

    /// Connects to `addr` (`host:port`), which must present a certificate for `server_name`.
    pub fn connect_with(addr: &str, server_name: &str, config: quinn::ClientConfig) -> StreamResult<Arc<Self>> {
        let socket: SocketAddr = resolve(addr)?;
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).thread_name("lucem-remote").enable_all().build()?;
        let session = runtime.block_on(Session::connect(socket, server_name, config))?;
        Ok(Arc::new(Self { session, runtime, addr: addr.to_string() }))
    }

    /// `host:port` of the server.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// The images the server offers (paths inside its data folder).
    pub fn list(&self) -> StreamResult<Vec<String>> {
        self.runtime.block_on(self.session.list())
    }

    /// One image's description.
    pub fn info(&self, path: &str) -> StreamResult<ImageInfo> {
        self.runtime.block_on(self.session.info(path))
    }

    /// Calls `method` of the server's plugin `plugin`.
    pub fn plugin_call(&self, plugin: &str, method: &str, body: serde_json::Value) -> StreamResult<serde_json::Value> {
        self.runtime.block_on(self.session.plugin_call(plugin, method, body))
    }

    /// Opens the served image `path` as a pyramid named `lucem://<addr>/<path>`.
    pub fn open(self: &Arc<Self>, path: &str) -> StreamResult<Pyramid> {
        let info = self.info(path)?;
        pyramid_of(self.clone(), path, &info)
    }

    /// Closes the connection (the server sees a clean goodbye).
    pub fn close(&self) {
        self.session.close();
    }
}

fn resolve(addr: &str) -> StreamResult<SocketAddr> {
    use std::net::ToSocketAddrs as _;
    addr.to_socket_addrs()?.next().ok_or_else(|| StreamError::protocol(format!("{addr}: no address")))
}

/// The pyramid a server describes, its tiles read through `remote`.
fn pyramid_of(remote: Arc<Remote>, path: &str, info: &ImageInfo) -> StreamResult<Pyramid> {
    let Some(first) = info.levels.first() else { return Err(StreamError::protocol("the image has no levels")) };
    let (d0, w0) = (first[0].max(1) as f64, first[2].max(1) as f64);
    let levels = info
        .levels
        .iter()
        .enumerate()
        .map(|(i, &[depth, height, width, tile_d, tile_h, tile_w])| LevelInfo {
            path: format!("{i}"),
            width,
            height,
            tile_w,
            tile_h,
            downsample: w0 / width.max(1) as f64,
            depth,
            tile_d: tile_d.max(1),
            z_downsample: d0 / depth.max(1) as f64,
        })
        .collect();
    let channels = info
        .channels
        .iter()
        .enumerate()
        .map(|(i, label)| match info.display.get(i) {
            Some(d) => ChannelInfo { label: label.clone(), color: d.color, window: d.window, range: d.range, active: d.active },
            None => ChannelInfo { label: label.clone(), color: [255, 255, 255], window: (0.0, 65535.0), range: (0.0, 65535.0), active: true },
        })
        .collect();
    let name = remote_name(&remote.addr, path);
    let source = Arc::new(RemoteTiles { remote, path: path.to_string() });
    Pyramid::from_source(info.name.clone(), Some(PathBuf::from(name)), levels, channels, info.pixel_size_um, info.z_spacing_um, info.dtype.clone(), source).map_err(|e| StreamError::protocol(e.to_string()))
}

struct RemoteTiles {
    remote: Arc<Remote>,
    path: String,
}

impl TileSource for RemoteTiles {
    fn tile(&self, level: usize, channel: usize, z: usize, tx: usize, ty: usize) -> Result<(usize, usize, Vec<f32>), String> {
        let t = self.remote.runtime.block_on(self.remote.session.tile(&self.path, level, channel, z, tx, ty)).map_err(|e| e.to_string())?;
        Ok((t.width, t.height, t.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_and_records_are_found() {
        assert_eq!(split_remote("lucem://10.0.0.5:50051/slides/a.ome.zarr"), Some(("10.0.0.5:50051", "slides/a.ome.zarr")));
        assert_eq!(split_remote("lucem://10.0.0.5:50051/"), None);
        assert_eq!(split_remote("/data/a.tif"), None);
        let dir = std::env::temp_dir().join(format!("lucem-remote-records-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("10.0.0.5_50051.pem"), "x").unwrap();
        std::fs::write(dir.join("old_7000.pem"), "x").unwrap();
        PairedServer { addr: "10.0.0.5:50051".into(), names: vec!["lab".into()], identity: dir.join("10.0.0.5_50051.pem") }.save().unwrap();
        let all = PairedServer::all_in(&dir);
        assert_eq!(all.len(), 2);
        assert_eq!((all[0].addr.as_str(), all[0].names[0].as_str()), ("10.0.0.5:50051", "lab"));
        assert_eq!((all[1].addr.as_str(), all[1].names[0].as_str()), ("old:7000", "localhost"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
