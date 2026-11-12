//! The client side of a session.

use std::net::SocketAddr;

use crate::error::{StreamError, StreamResult};
use crate::protocol::{read_message, write_message, ImageInfo, Request, Response, TileData, MAX_TILE_BYTES};

/// A connection to a server.
pub struct Session {
    _endpoint: quinn::Endpoint,
    connection: quinn::Connection,
}

impl Session {
    /// Connects to `addr`, which must present a certificate for `server_name`.
    pub async fn connect(addr: SocketAddr, server_name: &str, config: quinn::ClientConfig) -> StreamResult<Self> {
        let bind: SocketAddr = if addr.is_ipv6() { "[::]:0".parse().expect("an address") } else { "0.0.0.0:0".parse().expect("an address") };
        let mut endpoint = quinn::Endpoint::client(bind)?;
        endpoint.set_default_client_config(config);
        let connection = endpoint.connect(addr, server_name)?.await?;
        Ok(Self { _endpoint: endpoint, connection })
    }

    async fn ask(&self, request: &Request) -> StreamResult<(Response, quinn::RecvStream)> {
        let (mut send, mut recv) = self.connection.open_bi().await?;
        write_message(&mut send, request).await?;
        send.finish()?;
        let response: Response = read_message(&mut recv).await?;
        if let Response::Error { reason } = response {
            return Err(StreamError::Server(reason));
        }
        Ok((response, recv))
    }

    /// The images the server offers.
    pub async fn list(&self) -> StreamResult<Vec<String>> {
        match self.ask(&Request::List).await?.0 {
            Response::Images { paths } => Ok(paths),
            other => Err(StreamError::protocol(format!("unexpected answer {other:?}"))),
        }
    }

    pub async fn info(&self, path: &str) -> StreamResult<ImageInfo> {
        match self.ask(&Request::Info { path: path.into() }).await?.0 {
            Response::Info { info } => Ok(info),
            other => Err(StreamError::protocol(format!("unexpected answer {other:?}"))),
        }
    }

    pub async fn tile(&self, path: &str, level: usize, channel: usize, z: usize, tx: usize, ty: usize) -> StreamResult<TileData> {
        let (response, mut recv) = self.ask(&Request::Tile { path: path.into(), level, channel, z, tx, ty }).await?;
        let Response::Tile { width, height } = response else { return Err(StreamError::protocol("expected a tile")) };
        let bytes = recv.read_to_end(MAX_TILE_BYTES).await.map_err(StreamError::protocol)?;
        if width.checked_mul(height).and_then(|n| n.checked_mul(4)) != Some(bytes.len()) {
            return Err(StreamError::protocol("the tile's size does not match its data"));
        }
        let data = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok(TileData { width, height, data })
    }

    pub fn close(&self) {
        self.connection.close(quinn::VarInt::from_u32(0), b"done");
    }
}
