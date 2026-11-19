//! Data layer: the shared store (project document, undo, open images, tool state, results), the
//! GPU device, what is remembered between runs, and the result computation.

pub mod device;
pub mod document;
pub mod geojson;
pub mod hist;
pub mod jobs;
pub mod layout;
pub mod labels;
pub mod ml;
pub mod model_node;
pub mod prefs;
pub mod python_worker;
pub mod recipes;
pub mod remote;
pub mod results;
pub mod script;
pub mod selection;
pub mod settings;
pub mod store;
pub mod tools;
