//! Images on the servers this computer is paired with (`lucem pair`), opened as
//! `lucem://<host:port>/<path>`: the connection is made the first time a server is used and kept
//! for the session; its images are listed when asked (Connect to a lucem server…) and offered in
//! the server menu. A served image is a pyramid like a file's, its tiles read over the connection.

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

/// The connection to the paired server at `addr`, made if needed (blocking: call it off the UI
/// thread).
pub fn connection(addr: &str) -> Result<Arc<Remote>, String> {
    if let Some(r) = connections().lock().unwrap_or_else(|e| e.into_inner()).get(addr) {
        return Ok(r.clone());
    }
    let server = paired().into_iter().find(|s| s.addr == addr).ok_or_else(|| format!("this computer is not paired with {addr}: `lucem pair {addr} <code>` pairs it"))?;
    let remote = Remote::connect(&server).map_err(|e| format!("{addr}: {e}"))?;
    connections().lock().unwrap_or_else(|e| e.into_inner()).insert(addr.to_string(), remote.clone());
    Ok(remote)
}

/// Opens `lucem://<addr>/<path>` (blocking).
pub fn open(name: &str) -> Result<Pyramid, String> {
    let (addr, path) = lucem_stream::remote::split_remote(name).ok_or_else(|| format!("{name}: not a served image (lucem://host:port/path)"))?;
    let remote = connection(addr)?;
    match remote.open(path) {
        Ok(py) => Ok(py),
        Err(e) => {
            // A dropped connection is made again next time
            connections().lock().unwrap_or_else(|e| e.into_inner()).remove(addr);
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
                connections().lock().unwrap_or_else(|e| e.into_inner()).remove(&s.addr);
                format!("{}: {e}", s.addr)
            })
        });
        if let Ok(list) = &found {
            images += list.len();
            reached += 1;
        }
        listed().lock().unwrap_or_else(|e| e.into_inner()).insert(s.addr.clone(), found);
    }
    format!("{images} image(s) on {reached} of {} paired server(s): the server menu lists them", servers.len())
}

/// What the servers offered when last asked, by address.
pub fn last_listed() -> BTreeMap<String, Result<Vec<String>, String>> {
    listed().lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Whether a path names a served image.
pub fn is_remote(path: &std::path::Path) -> bool {
    path.to_string_lossy().starts_with(lucem_stream::REMOTE_SCHEME)
}
