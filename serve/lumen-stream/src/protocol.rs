//! The wire protocol: each request is one QUIC bidirectional stream carrying a JSON request; the
//! answer is a length-prefixed JSON response, followed for a tile by its values (little-endian
//! `f32`) to the end of the stream.

use serde::{Deserialize, Serialize};

use crate::error::{StreamError, StreamResult};

/// TLS application protocol (ALPN): names the protocol and its version, so QUIC refuses peers
/// speaking another.
pub const ALPN: &[u8] = b"lumen-stream/1";
/// Largest JSON message either side accepts.
pub const MAX_MESSAGE_BYTES: usize = 1 << 20;
/// Largest tile a client accepts.
pub const MAX_TILE_BYTES: usize = 1 << 30;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    /// The images under the server's data folder.
    List,
    /// One image's description (`path` relative to the data folder).
    Info { path: String },
    Tile { path: String, level: usize, channel: usize, z: usize, tx: usize, ty: usize },
    /// Pairing a computer ([`crate::pairing`]): the rest of the exchange follows on the stream.
    Pair { hello: crate::pairing::PairHello },
}

/// One image's description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInfo {
    pub name: String,
    pub dtype: String,
    pub channels: Vec<String>,
    pub pixel_size_um: Option<f64>,
    pub z_spacing_um: Option<f64>,
    /// `[depth, height, width, tile_d, tile_h, tile_w]` of each level, finest first.
    pub levels: Vec<[usize; 6]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "response", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Images { paths: Vec<String> },
    Info { info: ImageInfo },
    /// The tile's values follow.
    Tile { width: usize, height: usize },
    Error { reason: String },
}

/// A tile as received.
#[derive(Debug, Clone, PartialEq)]
pub struct TileData {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

pub(crate) async fn write_message<T: Serialize>(send: &mut quinn::SendStream, message: &T) -> StreamResult<()> {
    let body = serde_json::to_vec(message)?;
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(&body).await?;
    Ok(())
}

pub(crate) async fn read_message<T: for<'a> Deserialize<'a>>(recv: &mut quinn::RecvStream) -> StreamResult<T> {
    let mut len = [0u8; 4];
    recv.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_MESSAGE_BYTES {
        return Err(StreamError::TooLarge { len, max: MAX_MESSAGE_BYTES });
    }
    let mut body = vec![0u8; len];
    recv.read_exact(&mut body).await?;
    Ok(serde_json::from_slice(&body)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_with_unknown_fields_are_refused() {
        assert!(serde_json::from_str::<Request>(r#"{"request":"info","path":"a","x":1}"#).is_err());
        let t = serde_json::from_str::<Request>(r#"{"request":"tile","path":"a","level":0,"channel":0,"z":0,"tx":1,"ty":2}"#).unwrap();
        assert!(matches!(t, Request::Tile { tx: 1, ty: 2, .. }));
    }
}
