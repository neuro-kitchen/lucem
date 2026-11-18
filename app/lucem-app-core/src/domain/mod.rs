//! Plain models shared by the screens and the data layer: the project document, its graph, its
//! annotations, and the commands that change it ([`lucem_session`], re-exported here so the app's
//! paths stay `crate::domain::…`), and the app's own tools.

pub use lucem_session::{annotation, boolean, command, geometry, graph, hierarchy, paths, project};
pub mod tool;
