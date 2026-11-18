//! The wire protocol: each request is one QUIC bidirectional stream carrying a JSON request; the
//! answer is a length-prefixed JSON response, followed for a tile by its values (little-endian
//! `f32`) to the end of the stream.

use serde::{Deserialize, Serialize};

#[cfg(feature = "native")]
use crate::error::{StreamError, StreamResult};

/// TLS application protocol (ALPN): names the protocol and its version, so QUIC refuses peers
/// speaking another.
pub const ALPN: &[u8] = b"lucem-stream/1";
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
    Pair { hello: crate::pair::PairHello },
    /// A call to one of the server's plugins ([`crate::plugin`]): its id, the method, a JSON body.
    Plugin { plugin: String, method: String, body: serde_json::Value },
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
    /// How each channel is shown, as the file suggests (same order as `channels`; empty from an
    /// older server: defaults then).
    #[serde(default)]
    pub display: Vec<ChannelDisplay>,
}

/// One channel's suggested display: colour (RGB), window and range (raw values), shown at first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelDisplay {
    pub color: [u8; 3],
    pub window: (f32, f32),
    pub range: (f32, f32),
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "response", rename_all = "snake_case", deny_unknown_fields)]
pub enum Response {
    Images { paths: Vec<String> },
    Info { info: ImageInfo },
    /// The tile's values follow.
    Tile { width: usize, height: usize },
    Error { reason: String },
    /// A plugin's answer.
    Plugin { body: serde_json::Value },
}

/// A tile as received.
#[derive(Debug, Clone, PartialEq)]
pub struct TileData {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[cfg(feature = "native")]
pub(crate) async fn write_message<T: Serialize>(send: &mut quinn::SendStream, message: &T) -> StreamResult<()> {
    let body = serde_json::to_vec(message)?;
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(&body).await?;
    Ok(())
}

#[cfg(feature = "native")]
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

/// The levels and channels of a pyramid described by `info` (whoever has the image: a server
/// over QUIC, or a page's server over WebTransport), to open it with
/// [`lucem_io::Pyramid::from_source`].
pub fn pyramid_parts(info: &ImageInfo) -> (Vec<lucem_io::LevelInfo>, Vec<lucem_io::ChannelInfo>) {
    let first = info.levels.first().copied().unwrap_or([1, 1, 1, 1, 1, 1]);
    let (d0, w0) = (first[0].max(1) as f64, first[2].max(1) as f64);
    let levels = info
        .levels
        .iter()
        .enumerate()
        .map(|(i, &[depth, height, width, tile_d, tile_h, tile_w])| lucem_io::LevelInfo {
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
            Some(d) => lucem_io::ChannelInfo { label: label.clone(), color: d.color, window: d.window, range: d.range, active: d.active },
            None => lucem_io::ChannelInfo { label: label.clone(), color: [255, 255, 255], window: (0.0, 65535.0), range: (0.0, 65535.0), active: true },
        })
        .collect();
    (levels, channels)
}

/// How a served image's name starts: `lucem://<host:port>/<path inside the data folder>` (in a
/// page, the host is the page's server).
pub const REMOTE_SCHEME: &str = "lucem://";

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
