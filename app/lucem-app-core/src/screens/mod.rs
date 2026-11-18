//! The screens: home, Explore, Pipeline, Analyze, Settings' sections.

/// Tables (Polars): not in a browser build yet.
#[cfg(not(target_family = "wasm"))]
pub mod analyze;
pub mod explore;
pub mod home;
pub mod pipeline;
pub mod settings;
pub mod sign_in;
