//! Data layer: the shared store (project document, undo, open images, tool state, results), the
//! GPU device, what is remembered between runs, and the result computation.

pub mod device;
pub mod prefs;
pub mod results;
pub mod store;
