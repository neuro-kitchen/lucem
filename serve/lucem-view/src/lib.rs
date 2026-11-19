//! Viewing support for `lucem`: interactive viewport camera math (`zoom`, `pan`, `fit_to_viewport`,
//! `screen_to_image`, `image_to_screen`), colormaps, and BGRA `RenderImage` tonemapping.

pub mod tiles;
pub mod tonemap;
pub mod viewport;
pub mod volume;

pub use tonemap::{build_render_image, ColormapLut, TonemapSettings};
pub use viewport::ViewportCamera;
pub use volume::{project_point, ray_midpoint_frac, render_volume, render_volume_rgba, render_volume_rgba_clipped, render_volume_rgba_cpu, unpack_bgra, OrbitCamera, VolumeChannel, VolumeClip, VolumeMode, VolumeQuality};
pub use tiles::{bgra_image, composite_bgra, label_color, render_layer, ChannelDisplay, LayerStyle, TileImage, TileKey, RegionReader, TileResult, TileService};
