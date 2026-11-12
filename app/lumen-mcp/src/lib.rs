//! lumen for agents: the tools of the Model Context Protocol (MCP) over lumen without a window,
//! and the work they share with the command line.
//!
//! - [`work`]: the operations, running a pipeline, running a session on an image and writing its
//!   results. `lumen ops`, `lumen run` and `lumen session run` call these too.
//! - [`server`]: the tools ([`server::LumenTools`]) and [`server::serve_stdio`], which
//!   `lumen mcp` runs: an agent (Claude Code, or any MCP client) starts it and calls the tools.
//!
//! The app will host the same tools and add its own (annotations, display, views), so an agent
//! in its chat works on what the window shows.

pub mod server;
pub mod work;

pub use server::{serve_stdio, LumenTools};
