//! Images on the servers this computer is paired with (`lucem pair`), opened as
//! `lucem://<host:port>/<path>`: the connection is made the first time a server is used and kept
//! for the session; its images are listed when asked (Connect to a lucem server…) and offered in
//! the server menu. A served image is a pyramid like a file's, its tiles read over the connection.

//! In a browser page there is one server, the page's own (`lucem serve --web`): its connection
//! is set once the page signs in ([`set_page_link`]), and its images are `lucem://<its host>/…`.

pub use imp::*;

/// Whether a path names a served image.
pub fn is_remote(path: &std::path::Path) -> bool {
    path.to_string_lossy().starts_with(lucem_stream::REMOTE_SCHEME)
}

#[cfg(not(target_family = "wasm"))]
mod imp {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use lucem_io::Pyramid;
    use lucem_stream::{PairedServer, Remote};

    /// The connections made, by address.
    fn connections() -> &'static Mutex<BTreeMap<String, Arc<Remote>>> {
        static C: OnceLock<Mutex<BTreeMap<String, Arc<Remote>>>> = OnceLock::new();
        C.get_or_init(Default::default)
    }

    /// What each paired server offered when last asked: its images, or why it could not be reached.
    fn listed() -> &'static Mutex<BTreeMap<String, Result<Vec<String>, String>>> {
        static L: OnceLock<Mutex<BTreeMap<String, Result<Vec<String>, String>>>> = OnceLock::new();
        L.get_or_init(Default::default)
    }

    /// The paired servers.
    pub fn paired() -> Vec<PairedServer> {
        PairedServer::all()
    }

    /// Whether there is a server to ask (a paired one).
    pub fn has_servers() -> bool {
        !paired().is_empty()
    }

    /// The connection to the paired server at `addr`, made if needed (blocking: call it off the UI
    /// thread).
    pub fn connection(addr: &str) -> Result<Arc<Remote>, String> {
        if let Some(r) = crate::platform::lock(connections()).get(addr) {
            return Ok(r.clone());
        }
        let server = paired().into_iter().find(|s| s.addr == addr).ok_or_else(|| format!("this computer is not paired with {addr}: `lucem pair {addr} <code>` pairs it"))?;
        let remote = Remote::connect(&server).map_err(|e| format!("{addr}: {e}"))?;
        crate::platform::lock(connections()).insert(addr.to_string(), remote.clone());
        Ok(remote)
    }

    /// Opens `lucem://<addr>/<path>` (blocking).
    pub fn open(name: &str) -> Result<Pyramid, String> {
        let (addr, path) = lucem_stream::split_remote(name).ok_or_else(|| format!("{name}: not a served image (lucem://host:port/path)"))?;
        let remote = connection(addr)?;
        match remote.open(path) {
            Ok(py) => Ok(py),
            Err(e) => {
                // A dropped connection is made again next time
                crate::platform::lock(connections()).remove(addr);
                Err(format!("{name}: {e}"))
            }
        }
    }

    /// Asks every paired server for its images (blocking); a summary of what was found.
    pub fn refresh() -> String {
        let servers = paired();
        if servers.is_empty() {
            return "No paired servers: `lucem pair <host:port> <code>` pairs this computer with one".into();
        }
        let mut images = 0;
        let mut reached = 0;
        for s in &servers {
            let found = connection(&s.addr).and_then(|r| {
                r.list().map_err(|e| {
                    crate::platform::lock(connections()).remove(&s.addr);
                    format!("{}: {e}", s.addr)
                })
            });
            if let Ok(list) = &found {
                images += list.len();
                reached += 1;
            }
            crate::platform::lock(listed()).insert(s.addr.clone(), found);
        }
        format!("{images} image(s) on {reached} of {} paired server(s): the server menu lists them", servers.len())
    }

    /// What the servers offered when last asked, by address.
    pub fn last_listed() -> BTreeMap<String, Result<Vec<String>, String>> {
        crate::platform::lock(listed()).clone()
    }

}

#[cfg(target_family = "wasm")]
mod imp {
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

    /// Whether there is a server to ask (the page's, once signed in).
    pub fn has_servers() -> bool {
        LINK.get().is_some()
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
}
