//! Image file decoding and encoding into/from host and device buffers.

use std::path::{Path, PathBuf};

use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use crate::error::{IoError, IoResult};

/// A decoded image ready for GPU upload or display.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Source file path, or synthetic identifier for generated canvases/phantoms.
    pub name: String,
    pub path: Option<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    /// Original bit depth description (e.g. `"uint8"`, `"uint16"`, `"float32"`).
    pub dtype_label: String,
    /// Normalized `[0.0, 1.0]` luminance plane `[height, width]` in row-major order for single-channel
    /// GPU operators.
    pub luminance_f32: Vec<f32>,
    /// Interleaved RGBA `[height, width, 4]` in `[0.0, 1.0]` for color-space and stain ops.
    pub rgba_f32: Vec<f32>,
}

impl LoadedImage {
    /// Creates a new uniform blank canvas of `width x height` filled with `fill_intensity` in `[0.0, 1.0]`.
    pub fn blank_canvas(name: impl Into<String>, width: usize, height: usize, fill_intensity: f32) -> IoResult<Self> {
        if width == 0 || height == 0 {
            return Err(IoError::InvalidDimensions {
                width,
                height,
                reason: "canvas dimensions must be >= 1",
            });
        }
        let v = fill_intensity.clamp(0.0, 1.0);
        let n = width * height;
        let luminance_f32 = vec![v; n];
        let mut rgba_f32 = Vec::with_capacity(n * 4);
        for _ in 0..n {
            rgba_f32.extend_from_slice(&[v, v, v, 1.0]);
        }
        Ok(Self {
            name: name.into(),
            path: None,
            width,
            height,
            channels: 1,
            dtype_label: "float32".to_string(),
            luminance_f32,
            rgba_f32,
        })
    }

    /// Uploads the `[height, width]` normalized luminance plane to the active CubeCL [`Client`].
    pub fn to_device_luminance(&self, client: &Client) -> IoResult<DeviceArray<f32>> {
        Ok(DeviceArray::from_host(
            client,
            Shape::new([self.height, self.width]),
            self.luminance_f32.clone(),
        )?)
    }

    /// Uploads the `[height, width, 3]` RGB tensor to the active CubeCL [`Client`].
    pub fn to_device_rgb(&self, client: &Client) -> IoResult<DeviceArray<f32>> {
        let mut rgb = Vec::with_capacity(self.height * self.width * 3);
        for px in self.rgba_f32.chunks_exact(4) {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        Ok(DeviceArray::from_host(
            client,
            Shape::new([self.height, self.width, 3]),
            rgb,
        )?)
    }
}

/// Loads an image file or `.ome.zarr` pyramid directory from disk (`.ome.zarr`, `.png`, `.tif`,
/// `.tiff`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.gif`, `.hdr`, `.exr`).
pub fn load_image_file(path: impl AsRef<Path>) -> IoResult<LoadedImage> {
    let path_ref = path.as_ref();
    if crate::ome_zarr::is_ome_zarr_path(path_ref) {
        return crate::ome_zarr::load_ome_zarr(path_ref);
    }
    let dyn_img = image::ImageReader::open(path_ref)?
        .with_guessed_format()?
        .decode()?;

    let width = dyn_img.width() as usize;
    let height = dyn_img.height() as usize;
    if width == 0 || height == 0 {
        return Err(IoError::InvalidDimensions {
            width,
            height,
            reason: "decoded image has zero width or height",
        });
    }

    let (channels, dtype_label) = match &dyn_img {
        image::DynamicImage::ImageLuma8(_) => (1, "uint8"),
        image::DynamicImage::ImageLumaA8(_) => (2, "uint8"),
        image::DynamicImage::ImageRgb8(_) => (3, "uint8"),
        image::DynamicImage::ImageRgba8(_) => (4, "uint8"),
        image::DynamicImage::ImageLuma16(_) => (1, "uint16"),
        image::DynamicImage::ImageLumaA16(_) => (2, "uint16"),
        image::DynamicImage::ImageRgb16(_) => (3, "uint16"),
        image::DynamicImage::ImageRgba16(_) => (4, "uint16"),
        image::DynamicImage::ImageRgb32F(_) => (3, "float32"),
        image::DynamicImage::ImageRgba32F(_) => (4, "float32"),
        _ => (4, "uint8"),
    };

    let rgba32f = dyn_img.to_rgba32f();
    let raw = rgba32f.as_raw();
    let n = width * height;
    let mut luminance_f32 = Vec::with_capacity(n);
    let mut rgba_f32 = Vec::with_capacity(n * 4);

    for px in raw.chunks_exact(4) {
        let r = px[0].clamp(0.0, 1.0);
        let g = px[1].clamp(0.0, 1.0);
        let b = px[2].clamp(0.0, 1.0);
        let a = px[3].clamp(0.0, 1.0);
        // ITU-R BT.709 luminance
        let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0);
        luminance_f32.push(lum);
        rgba_f32.extend_from_slice(&[r, g, b, a]);
    }

    let name = path_ref
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_string();

    Ok(LoadedImage {
        name,
        path: Some(path_ref.to_path_buf()),
        width,
        height,
        channels,
        dtype_label: dtype_label.to_string(),
        luminance_f32,
        rgba_f32,
    })
}

/// Saves a `[height, width]` normalized `f32` buffer to disk as an 8-bit grayscale PNG/TIFF/JPEG.
pub fn save_image_file(
    path: impl AsRef<Path>,
    width: usize,
    height: usize,
    luminance_f32: &[f32],
) -> IoResult<()> {
    if width * height != luminance_f32.len() {
        return Err(IoError::InvalidDimensions {
            width,
            height,
            reason: "buffer length does not match width * height",
        });
    }
    let bytes: Vec<u8> = luminance_f32
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::save_buffer(
        path,
        &bytes,
        width as u32,
        height as u32,
        image::ExtendedColorType::L8,
    )?;
    Ok(())
}
