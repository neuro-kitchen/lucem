//! Tonemapping and scientific false-color LUT rendering into `gpui_kit::RenderImage` (BGRA8).

use std::sync::Arc;

use gpui_kit::RenderImage;
use image::{Frame, ImageBuffer, Rgba};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// False-color lookup tables for scientific and creative inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColormapLut {
    Grayscale,
    CyanGfp,
    Viridis,
    Magma,
    Inferno,
}

impl ColormapLut {
    pub const ALL: &'static [Self] = &[
        Self::Grayscale,
        Self::CyanGfp,
        Self::Viridis,
        Self::Magma,
        Self::Inferno,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Grayscale => "Grayscale",
            Self::CyanGfp => "GFP / Cyan",
            Self::Viridis => "Viridis",
            Self::Magma => "Magma",
            Self::Inferno => "Inferno",
        }
    }

    /// Maps a normalized scalar `t` in `[0.0, 1.0]` to `[R, G, B]` in `0..=255`.
    #[inline]
    pub fn sample_rgb(self, t: f32) -> [u8; 3] {
        let u = t.clamp(0.0, 1.0);
        let (r, g, b) = match self {
            Self::Grayscale => (u, u, u),
            Self::CyanGfp => (
                (u * u * 0.35).clamp(0.0, 1.0),
                (u.powf(0.85) * 0.98).clamp(0.0, 1.0),
                (u.powf(0.75) * 0.88).clamp(0.0, 1.0),
            ),
            Self::Viridis => {
                // Smooth polynomial approximation of Viridis
                let r = (0.267 + u * (-0.004 + u * (1.28 + u * (-1.85 + u * 1.30)))).clamp(0.0, 1.0);
                let g = (0.005 + u * (1.01 + u * (-0.18 + u * 0.08))).clamp(0.0, 1.0);
                let b = (0.329 + u * (1.16 + u * (-2.42 + u * 1.08))).clamp(0.0, 1.0);
                (r, g, b)
            }
            Self::Magma => {
                let r = (u.powf(0.75) * 1.05).clamp(0.0, 1.0);
                let g = (u * u * (1.15 - 0.18 * u)).clamp(0.0, 1.0);
                let b = ((0.15 + 1.8 * u - 1.95 * u * u + 0.95 * u * u * u) * u.sqrt()).clamp(0.0, 1.0);
                (r, g, b)
            }
            Self::Inferno => {
                let r = (1.18 * u.powf(0.68)).clamp(0.0, 1.0);
                let g = (u * u * (1.35 - 0.38 * u)).clamp(0.0, 1.0);
                let b = (1.6 * u * (1.0 - u) * (1.0 - 0.4 * u)).clamp(0.0, 1.0);
                (r, g, b)
            }
        };
        [
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
        ]
    }
}

/// Display exposure, window/level, and LUT settings applied when converting a float buffer into a
/// viewport `RenderImage`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TonemapSettings {
    pub colormap: ColormapLut,
    pub exposure: f32,
    pub gamma: f32,
    pub black_point: f32,
    pub white_point: f32,
    pub invert: bool,
}

impl Default for TonemapSettings {
    fn default() -> Self {
        Self {
            colormap: ColormapLut::Grayscale,
            exposure: 1.0,
            gamma: 1.0,
            black_point: 0.0,
            white_point: 1.0,
            invert: false,
        }
    }
}

/// Converts a `[height, width]` `f32` buffer (plus optional Marching Squares contour points) into an
/// `Arc<RenderImage>` in GPUI's native BGRA8 layout.
pub fn build_render_image(
    width: usize,
    height: usize,
    luminance: &[f32],
    settings: &TonemapSettings,
    contours: Option<&[Vec<[f64; 2]>]>,
) -> Arc<RenderImage> {
    let w = width.max(1) as u32;
    let h = height.max(1) as u32;
    let mut raw_bgra = vec![0u8; (w as usize) * (h as usize) * 4];

    let span = (settings.white_point - settings.black_point).max(1e-5);
    let inv_gamma = 1.0 / settings.gamma.max(0.1);

    for (idx, &val) in luminance.iter().take((w as usize) * (h as usize)).enumerate() {
        let mut t = ((val - settings.black_point) / span) * settings.exposure;
        t = t.clamp(0.0, 1.0);
        if (inv_gamma - 1.0).abs() > 1e-3 {
            t = t.powf(inv_gamma);
        }
        if settings.invert {
            t = 1.0 - t;
        }
        let [r, g, b] = settings.colormap.sample_rgb(t);
        // GPUI's RenderImage stores BGRA8 in its Frame buffer
        let base = idx * 4;
        raw_bgra[base] = b;
        raw_bgra[base + 1] = g;
        raw_bgra[base + 2] = r;
        raw_bgra[base + 3] = 255;
    }

    // Burn VectorCraft-style Marching Squares iso-contours directly into the display frame when enabled
    if let Some(contour_list) = contours {
        let w_usize = w as usize;
        let h_usize = h as usize;
        for chain in contour_list {
            for seg in chain.windows(2) {
                let (y0, x0) = (seg[0][0], seg[0][1]);
                let (y1, x1) = (seg[1][0], seg[1][1]);
                let steps = ((y1 - y0).hypot(x1 - x0) * 2.0).ceil().max(1.0) as usize;
                for s in 0..=steps {
                    let u = (s as f64) / (steps as f64);
                    let py = (y0 + u * (y1 - y0)).round() as isize;
                    let px_i = (x0 + u * (x1 - x0)).round() as isize;
                    if py >= 0 && py < h_usize as isize && px_i >= 0 && px_i < w_usize as isize {
                        let base = ((py as usize) * w_usize + (px_i as usize)) * 4;
                        // Electric cyan/emerald contour highlight (BGRA)
                        raw_bgra[base] = 230;
                        raw_bgra[base + 1] = 245;
                        raw_bgra[base + 2] = 45;
                        raw_bgra[base + 3] = 255;
                    }
                }
            }
        }
    }

    let img_buf: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, raw_bgra).unwrap_or_else(|| ImageBuffer::from_pixel(1, 1, Rgba([0, 0, 0, 255])));
    let frame = Frame::new(img_buf);
    Arc::new(RenderImage::new(SmallVec::from_elem(frame, 1)))
}
