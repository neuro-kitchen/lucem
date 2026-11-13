//! Errors of a stream session.

use std::fmt;

/// What went wrong in a stream session.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("TLS: {0}")]
    Tls(String),
    #[error("connect: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("connection: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("read: {0}")]
    Read(#[from] quinn::ReadExactError),
    #[error("read: {0}")]
    ReadToEnd(#[from] quinn::ReadToEndError),
    #[error("write: {0}")]
    Write(#[from] quinn::WriteError),
    #[error("stream closed: {0}")]
    Closed(#[from] quinn::ClosedStream),
    #[error("message of {len} bytes exceeds the {max}-byte limit")]
    TooLarge { len: usize, max: usize },
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The peer sent something the protocol does not allow.
    #[error("protocol: {0}")]
    Protocol(String),
    /// The server refused or failed a request, and said why.
    #[error("server: {0}")]
    Server(String),
}

impl StreamError {
    pub(crate) fn protocol(message: impl fmt::Display) -> Self {
        Self::Protocol(message.to_string())
    }
}

pub type StreamResult<T> = Result<T, StreamError>;
