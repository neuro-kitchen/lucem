//! Images on the servers this computer is paired with (`lucem pair`), opened as
//! `lucem://<host:port>/<path>`: the connection is made the first time a server is used and kept
//! for the session; its images are listed when asked (Connect to a lucem server…) and offered in
//! the server menu. A served image is a pyramid like a file's, its tiles read over the connection.

//! In a browser page there is one server, the page's own (`lucem serve --web`): its connection
//! is set once the page signs in ([`set_page_link`]), and its images are `lucem://<its host>/…`.

/// This target's: `crate::platform::remote` (desktop or page).
pub use crate::platform::remote::*;

/// Whether a path names a served image.
pub fn is_remote(path: &std::path::Path) -> bool {
    path.to_string_lossy().starts_with(lucem_stream::REMOTE_SCHEME)
}
