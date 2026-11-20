//! The screens: home, Explore, Pipeline, Analyze, Settings' sections.

/// Tables (Polars) on desktop; server-backed table curation on wasm32.
#[cfg(not(target_family = "wasm"))]
pub mod analyze;
#[cfg(target_family = "wasm")]
pub mod analyze_web;
pub mod code;
pub mod explore;
pub mod home;
pub mod pipeline;
pub mod server_images;
pub mod settings;
pub mod sign_in;
