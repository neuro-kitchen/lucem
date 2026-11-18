//! The project document of lucem: the whole working project as one versioned JSON file, its
//! processing graph, its annotations and classes, and the commands that change it. Plain data: the
//! app, the command line and an agent (MCP) read and change the same thing through
//! [`command::Command`]s. Nothing here draws (the app's tools and icons stay in the app).

pub mod annotation;
pub mod boolean;
pub mod command;
pub mod geometry;
pub mod graph;
pub mod hierarchy;
pub mod paths;
pub mod project;
