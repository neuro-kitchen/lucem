//! Tables for lucem, as [Polars](https://pola.rs) data frames.
//!
//! This crate is the one place that depends on Polars: the other crates use the re-export
//! ([`polars`], [`prelude`]), so the whole workspace builds one Polars with one feature set, and
//! a browser build can leave it out in one place (Polars does not build for wasm32 since 0.54).
//! Domain tables are built on it in their own crates (measurements in `lucem-hist`, analytics in
//! `lucem-ops`).

#[cfg(feature = "polars")]
pub use polars;
#[cfg(feature = "polars")]
pub use polars::prelude;

#[cfg(feature = "polars")]
pub mod view;
