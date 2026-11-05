//! Image readers, writers, procedural phantoms, and project manifest I/O for `lumen`.

pub mod error;
pub mod image_io;
pub mod ome_zarr;
pub mod phantoms;
pub mod project_io;

pub use error::{IoError, IoResult};
pub use image_io::{load_image_file, save_image_file, LoadedImage};
pub use ome_zarr::{
    discover_workspace_data_paths, find_ome_zarr_root, inspect_ome_zarr_container,
    is_ome_zarr_path, load_ome_zarr, load_ome_zarr_with_options, OmeZarrContainerInfo,
};
pub use phantoms::{generate_phantom, PhantomKind};
pub use project_io::{ProjectManifest, SavedModifier};
