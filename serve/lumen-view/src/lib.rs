//! Viewing support for `lumen`: interactive viewport camera math (`zoom`, `pan`, `fit_to_viewport`,
//! `screen_to_image`, `image_to_screen`), colormaps, and BGRA `RenderImage` tonemapping.

pub mod tiles;
pub mod tonemap;
pub mod viewport;
pub mod volume;

pub use tonemap::{build_render_image, ColormapLut, TonemapSettings};
pub use viewport::ViewportCamera;
pub use volume::{render_volume, OrbitCamera, VolumeMode};
pub use tiles::{bgra_image, composite_bgra, label_color, render_layer, ChannelDisplay, LayerStyle, TileImage, TileKey, TileResult, TileService};
