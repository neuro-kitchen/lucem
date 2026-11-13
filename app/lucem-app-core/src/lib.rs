//! Lucem app core: the workstation (on kitchen-shell's frame: title bar, workspaces, menus,
//! search, status bar, Settings), the project document and its commands, the image canvas and
//! tools. The `lucem-app` binary is this core.
//!
//! Layout (as dsp-kitchen's app core): `data` (store, device, results), `domain` (plain models:
//! project, graph, annotations, commands, tools), `commands` (what the app can do, declared once
//! for menus, keys, search and agents), `navigation` (workspaces, keys),
//! `screens/<feature>`, `ui` (theme tokens, widgets).

pub mod app;
/// What a computer kept under lucem's old name (lumen) keeps working.
pub use lucem_ml::legacy;
pub mod commands;
pub mod data;
pub mod domain;
pub mod navigation;
pub mod screens;
pub mod ui;
