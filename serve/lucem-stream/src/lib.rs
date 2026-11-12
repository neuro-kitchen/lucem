//! Serving images over QUIC with TLS 1.3, to enrolled clients only.
//!
//! A server keeps an authority ([`auth::Authority`]) and accepts only clients it enrolled (mutual
//! TLS; the registry is read at every connection, so revoking takes effect at once). Clients list
//! the images under the server's data folder and read tiles of them, each request on its own QUIC
//! stream. Ported from dsp-kitchen's `dspk-stream` (authority, TLS, accept loop).

pub mod auth;
pub mod client;
pub mod error;
pub mod pairing;
pub mod protocol;
pub mod server;
pub mod tls;

pub use auth::{Authority, ClientIdentity, ClientRecord, Enrollment, Peer, Permission};
pub use client::Session;
pub use error::{StreamError, StreamResult};
pub use pairing::{pair, PairingCode, Paired};
pub use protocol::{ImageInfo, Request, Response, TileData};
pub use server::{ServeOptions, Server, DEFAULT_MAX_CONNECTIONS, REFUSED_CLIENT};
pub use tls::{client_config, client_config_with, server_config, server_config_with_clients, ServerIdentity, ServerTls, ServerTrust};
