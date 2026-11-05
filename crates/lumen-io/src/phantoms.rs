//! Built-in procedural scientific & creative phantoms for instant startup exploration.
//!
//! Lets the user test ArtCraft raster layers, VectorCraft contour tracing, and Blender-style
//! GPU modifier stacks immediately from the Startup Project Hub without needing an external file.

use std::f32::consts::PI;

use crate::image_io::LoadedImage;

/// Built-in synthetic datasets available from the Startup Project Hub and Left Option Rail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhantomKind {
    /// Confocal fluorescence brain-section nuclei & neurite filaments (`512 x 512`).
    FluorescenceNuclei,
    /// Siemens star + resolution bar target for filter, deconvolution & edge-detection testing (`512 x 512`).
    SiemensStarTarget,
    /// VectorCraft geometric shapes, rings & SDF contours for Marching Squares / active contours (`512 x 512`).
    VectorContourShapes,
    /// Smooth multi-frequency interference & phase fringes for restoration / unwrapping (`512 x 512`).
    InterferenceFringes,
}

impl PhantomKind {
    pub const ALL: &'static [Self] = &[
        Self::FluorescenceNuclei,
        Self::SiemensStarTarget,
        Self::VectorContourShapes,
        Self::InterferenceFringes,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::FluorescenceNuclei => "Fluorescence Brain Nuclei",
            Self::SiemensStarTarget => "Siemens Optical Resolution Star",
            Self::VectorContourShapes => "Vector & Morphometry Contours",
            Self::InterferenceFringes => "Holographic Phase Fringes",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::FluorescenceNuclei => "512×512 · Confocal soma + neurite filaments for Otsu, Watershed & RegionProps",
            Self::SiemensStarTarget => "512×512 · Radial frequency target for Gaussian, Canny, Wiener & TV Denoising",
            Self::VectorContourShapes => "512×512 · Multi-scale rings, stars & polygons for Marching Squares & Snakes",
            Self::InterferenceFringes => "512×512 · Modulated sinusoidal carrier with synthetic background illumination",
        }
    }

    pub fn badge(self) -> &'static str {
        match self {
            Self::FluorescenceNuclei => "BIO / SEGMENT",
            Self::SiemensStarTarget => "OPTICS / RESTORE",
            Self::VectorContourShapes => "VECTOR / CONTOUR",
            Self::InterferenceFringes => "PHASE / FILTER",
        }
    }
}

/// Generates a `width x height` synthetic [`LoadedImage`] for `kind`.
pub fn generate_phantom(kind: PhantomKind, width: usize, height: usize) -> LoadedImage {
    let w = width.max(64);
    let h = height.max(64);
    let n = w * h;
    let mut luminance_f32 = vec![0.0f32; n];
    let mut rgba_f32 = vec![0.0f32; n * 4];

    match kind {
        PhantomKind::FluorescenceNuclei => {
            // Deterministic pseudo-random cell centers (cy, cx, radius, peak_intensity, has_nucleolus)
            let cells: [(f32, f32, f32, f32, bool); 22] = [
                (0.18, 0.22, 0.048, 0.92, true),
                (0.26, 0.42, 0.038, 0.84, false),
                (0.16, 0.68, 0.052, 0.96, true),
                (0.32, 0.80, 0.042, 0.78, false),
                (0.45, 0.19, 0.044, 0.88, true),
                (0.48, 0.49, 0.058, 0.98, true),
                (0.39, 0.62, 0.035, 0.75, false),
                (0.58, 0.34, 0.046, 0.90, true),
                (0.64, 0.73, 0.050, 0.94, true),
                (0.76, 0.24, 0.041, 0.82, false),
                (0.81, 0.54, 0.049, 0.91, true),
                (0.79, 0.83, 0.039, 0.86, false),
                (0.28, 0.14, 0.031, 0.71, false),
                (0.12, 0.48, 0.034, 0.76, false),
                (0.66, 0.15, 0.036, 0.79, false),
                (0.88, 0.38, 0.033, 0.74, false),
                (0.53, 0.86, 0.037, 0.81, false),
                (0.35, 0.31, 0.029, 0.69, false),
                (0.69, 0.46, 0.032, 0.77, false),
                (0.22, 0.87, 0.030, 0.72, false),
                (0.87, 0.70, 0.035, 0.80, false),
                (0.55, 0.60, 0.028, 0.73, false),
            ];

            for y in 0..h {
                let ny = y as f32 / h as f32;
                for x in 0..w {
                    let nx = x as f32 / w as f32;
                    // Subtle uneven illumination background + fine hash noise
                    let bg = 0.04 + 0.03 * (-((ny - 0.5).powi(2) + (nx - 0.5).powi(2))).exp();
                    let hash = (((y * 73856093) ^ (x * 19349663)) & 0xFF) as f32 / 255.0;
                    let mut val = bg + 0.025 * (hash - 0.5);

                    // Fine dendritic / neurite ridges
                    let wave1 = ((nx * 14.0 + (ny * 6.0).sin() * 1.4).sin()).abs();
                    let filament = (-((wave1 - 0.08).max(0.0) * 32.0).powi(2)).exp() * 0.18;
                    val += filament;

                    for &(cy, cx, rad, peak, nucleolus) in &cells {
                        let dy = (ny - cy) / rad;
                        let dx = (nx - cx) / (rad * 0.88);
                        let r2 = dy * dy + dx * dx;
                        if r2 < 3.5 {
                            let mut soma = peak * (-0.65 * r2).exp();
                            if nucleolus {
                                let nd2 = ((ny - (cy - 0.008)) / (rad * 0.25)).powi(2)
                                    + ((nx - (cx + 0.008)) / (rad * 0.25)).powi(2);
                                soma += 0.22 * (-nd2).exp();
                            }
                            val = val.max(soma + bg);
                        }
                    }

                    let v = val.clamp(0.0, 1.0);
                    let idx = y * w + x;
                    luminance_f32[idx] = v;
                    rgba_f32[idx * 4] = (v * 0.25).clamp(0.0, 1.0);
                    rgba_f32[idx * 4 + 1] = (v * 0.95).clamp(0.0, 1.0);
                    rgba_f32[idx * 4 + 2] = (v * 0.85).clamp(0.0, 1.0);
                    rgba_f32[idx * 4 + 3] = 1.0;
                }
            }
        }
        PhantomKind::SiemensStarTarget => {
            let spokes = 24.0f32;
            for y in 0..h {
                let dy = (y as f32 / h as f32) - 0.5;
                for x in 0..w {
                    let dx = (x as f32 / w as f32) - 0.5;
                    let r = (dx * dx + dy * dy).sqrt();
                    let theta = dy.atan2(dx);
                    let v = if r < 0.025 {
                        0.95
                    } else if r < 0.38 {
                        let carrier = (spokes * theta).cos();
                        let contrast = (1.0 - (0.04 / r.max(0.04)).powi(2)).clamp(0.15, 1.0);
                        0.5 + 0.42 * contrast * (carrier * 4.0).clamp(-1.0, 1.0)
                    } else if r < 0.43 {
                        // Concentric calibration ring
                        if ((r - 0.38) * 120.0) as i32 % 2 == 0 {
                            0.85
                        } else {
                            0.12
                        }
                    } else {
                        0.08
                    };
                    let idx = y * w + x;
                    let c = v.clamp(0.0, 1.0);
                    luminance_f32[idx] = c;
                    rgba_f32[idx * 4] = c;
                    rgba_f32[idx * 4 + 1] = c;
                    rgba_f32[idx * 4 + 2] = c;
                    rgba_f32[idx * 4 + 3] = 1.0;
                }
            }
        }
        PhantomKind::VectorContourShapes => {
            for y in 0..h {
                let ny = y as f32 / h as f32;
                for x in 0..w {
                    let nx = x as f32 / w as f32;
                    let mut v = 0.06f32;

                    // Outer rounded frame
                    let d_ring = (((ny - 0.34).powi(2) + (nx - 0.34).powi(2)).sqrt() - 0.16).abs();
                    if d_ring < 0.028 {
                        v = v.max(0.88);
                    }

                    // 5-pointed star in lower-right
                    let sy = ny - 0.66;
                    let sx = nx - 0.66;
                    let sr = (sy * sy + sx * sx).sqrt();
                    let sa = sy.atan2(sx);
                    let star_rad = 0.14 + 0.055 * (5.0 * sa).cos();
                    if sr < star_rad {
                        v = v.max(0.94 * (1.0 - 0.3 * (sr / star_rad)));
                    }

                    //Rotated capsule in upper-right
                    let ry = (ny - 0.28) * 0.707 - (nx - 0.72) * 0.707;
                    let rx = (ny - 0.28) * 0.707 + (nx - 0.72) * 0.707;
                    let cx_clamped = rx.clamp(-0.09, 0.09);
                    let cap_d = ((ry).powi(2) + (rx - cx_clamped).powi(2)).sqrt();
                    if cap_d < 0.045 {
                        v = v.max(0.76);
                    }

                    // Nested target in lower-left
                    let td = ((ny - 0.72).powi(2) + (nx - 0.28).powi(2)).sqrt();
                    if td < 0.15 {
                        let ring = ((td * 45.0).sin() * 0.5 + 0.5) * 0.85;
                        v = v.max(ring);
                    }

                    let idx = y * w + x;
                    let c = v.clamp(0.0, 1.0);
                    luminance_f32[idx] = c;
                    rgba_f32[idx * 4] = c;
                    rgba_f32[idx * 4 + 1] = c;
                    rgba_f32[idx * 4 + 2] = c;
                    rgba_f32[idx * 4 + 3] = 1.0;
                }
            }
        }
        PhantomKind::InterferenceFringes => {
            for y in 0..h {
                let ny = (y as f32 / h as f32) - 0.5;
                for x in 0..w {
                    let nx = (x as f32 / w as f32) - 0.5;
                    let r2 = nx * nx + ny * ny;
                    let phase = 18.0 * PI * r2 + 6.0 * PI * nx + 2.5 * (4.0 * PI * ny).sin();
                    let envelope = (-2.2 * r2).exp();
                    let bg_ramp = 0.25 + 0.2 * (nx + 0.5) + 0.15 * (ny + 0.5);
                    let c = (bg_ramp + 0.35 * envelope * phase.cos()).clamp(0.0, 1.0);
                    let idx = y * w + x;
                    luminance_f32[idx] = c;
                    rgba_f32[idx * 4] = c;
                    rgba_f32[idx * 4 + 1] = c;
                    rgba_f32[idx * 4 + 2] = c;
                    rgba_f32[idx * 4 + 3] = 1.0;
                }
            }
        }
    }

    LoadedImage {
        name: kind.title().to_string(),
        path: None,
        width: w,
        height: h,
        channels: 1,
        dtype_label: "float32 (synthetic)".to_string(),
        luminance_f32,
        rgba_f32,
    }
}
