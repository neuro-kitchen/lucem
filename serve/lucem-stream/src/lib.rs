//! Serving images: to computers over QUIC with TLS 1.3 (enrolled clients only), and to browser
//! pages over WebTransport, or a WebSocket where UDP does not get through (paired pages, by
//! token).
//!
//! A server keeps an authority ([`auth::Authority`]) and lets in only clients it enrolled (mutual
//! TLS; the registry is read at every connection, so revoking takes effect at once) or pages it
//! paired. Clients list the images under the server's data folder and read tiles of them, each
//! request on its own QUIC stream; what answers them is [`engine`], whatever the transport.
//! Ported from dsp-kitchen's `dspk-stream` (authority, TLS, pairing, the browser endpoint).
//!
//! Features: `native` (default: servers and computers), `web` (a page's side, built for
//! WebAssembly: [`pair`], [`wire`], [`protocol`] and the page's connection).

pub mod error;
pub mod pair;
pub mod protocol;
pub mod wire;

#[cfg(feature = "native")]
pub mod auth;
#[cfg(feature = "native")]
pub mod client;
#[cfg(feature = "native")]
pub mod engine;
#[cfg(feature = "native")]
pub mod pairing;
#[cfg(feature = "native")]
pub mod plugin;
#[cfg(feature = "native")]
pub mod remote;
#[cfg(feature = "native")]
pub mod server;
#[cfg(feature = "native")]
pub mod tls;
#[cfg(feature = "native")]
pub mod web;
/// A browser page's connection to its server.
#[cfg(all(feature = "web", target_family = "wasm"))]
pub mod web_client;

pub use error::{StreamError, StreamResult};
pub use pair::{Permission, ServeInfo, WebWelcome};
pub use protocol::{remote_name, split_remote, ChannelDisplay, ImageInfo, Request, Response, TileData, REMOTE_SCHEME};

#[cfg(feature = "native")]
pub use auth::{Authority, ClientIdentity, ClientRecord, Enrollment, Peer};
#[cfg(feature = "native")]
pub use client::Session;
#[cfg(feature = "native")]
pub use pairing::{pair, PairingCode, Paired};
#[cfg(feature = "native")]
pub use remote::{PairedServer, Remote};
#[cfg(feature = "native")]
pub use server::{ServeOptions, Server, DEFAULT_MAX_CONNECTIONS, REFUSED_CLIENT};
#[cfg(feature = "native")]
pub use tls::{client_config, client_config_with, server_config, server_config_with_clients, ServerIdentity, ServerTls, ServerTrust};
#[cfg(feature = "native")]
pub use web::{WebOptions, WebServer, WebTls};
