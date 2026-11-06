//! Lumen app core: the workstation shell (title bar, workspaces, status bar), the project
//! document and its commands, the image canvas and tools. The `lumen-app` binary is this core.
//!
//! Layout (as dsp-kitchen's app core): `data` (store, device, results), `domain` (plain models:
//! project, graph, annotations, commands, tools), `navigation` (workspaces, actions),
//! `screens/<feature>`, `ui` (theme tokens, widgets).

pub mod app;
pub mod data;
pub mod domain;
pub mod navigation;
pub mod screens;
pub mod ui;
