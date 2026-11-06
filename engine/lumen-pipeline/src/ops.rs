//! Operators as data: each [`Op`] is one node of a pipeline graph, serialized as
//! `{"op": "gaussian_blur", "sigma": 2.0}`. An op has typed inputs ([`Op::inputs`]) and one typed
//! output ([`Op::output`]), so the graph can check a connection before it runs; it runs on the
//! device ([`Op::run`]) and says how many pixels of context it reads around a pixel ([`Op::halo`]),
//! so a region can be processed with the margin it needs.

use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_ops::core::{elementwise, StructuringElement};
use lumen_ops::feature::{canny, Canny};
use lumen_ops::filter::gaussian::{difference_of_gaussians, gaussian_blur, unsharp_mask, GaussianBlur};
use lumen_ops::filter::gradient::{laplace, sobel, GradientFilter};
use lumen_ops::filter::order::{median_filter, OrderFilter};
use lumen_ops::intensity::contrast::adjust_gamma;
use lumen_ops::intensity::equalize::{clahe, ClaheSettings};
use lumen_ops::measure::regions::{label_components, region_features, RegionFeatures};
use lumen_ops::morphology::binary::{binary_close, binary_dilate, binary_erode, binary_open, BinaryMorphology};
use lumen_ops::morphology::geodesic::remove_small_objects;
use lumen_ops::morphology::grayscale::{morph_gradient, white_tophat, GrayMorphology};
use lumen_ops::restore::background::{rolling_ball, RollingBall};
use lumen_ops::restore::denoise::{tv_chambolle, TvChambolle};
use lumen_ops::segment::watershed::{watershed, WatershedOptions};
use lumen_ops::threshold::global::otsu;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// What flows along an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PortType {
    /// Intensities (`f32`, one channel).
    Image,
    /// 0 / 1 per pixel.
    Mask,
    /// An object id per pixel (0: background).
    Labels,
    /// One row per object (measurements).
    Table,
}

impl PortType {
    pub fn name(self) -> &'static str {
        match self {
            PortType::Image => "image",
            PortType::Mask => "mask",
            PortType::Labels => "labels",
            PortType::Table => "table",
        }
    }
}

/// A value on the device (or, for tables, on the host).
#[derive(Clone)]
pub enum Value {
    Image(DeviceArray<f32>),
    Mask(DeviceArray<u32>),
    Labels(DeviceArray<u32>),
    Table(Vec<RegionFeatures>),
}

impl Value {
    pub fn port(&self) -> PortType {
        match self {
            Value::Image(_) => PortType::Image,
            Value::Mask(_) => PortType::Mask,
            Value::Labels(_) => PortType::Labels,
            Value::Table(_) => PortType::Table,
        }
    }

    /// `(height, width)` of a raster value.
    pub fn dims(&self) -> Option<(usize, usize)> {
        let d = match self {
            Value::Image(a) => a.dims().to_vec(),
            Value::Mask(a) | Value::Labels(a) => a.dims().to_vec(),
            Value::Table(_) => return None,
        };
        (d.len() == 2).then(|| (d[0], d[1]))
    }

    /// A raster value's pixels as `f32` on the host (labels as their ids, masks 0 / 1).
    pub fn to_host(&self, client: &Client) -> Option<Vec<f32>> {
        match self {
            Value::Image(a) => Some(a.to_host(client)),
            Value::Mask(a) | Value::Labels(a) => Some(a.to_host(client).into_iter().map(|v| v as f32).collect()),
            Value::Table(_) => None,
        }
    }
}

/// One operator and its parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    /// Gaussian smoothing.
    GaussianBlur { sigma: f64 },
    /// Median in a disk of `radius` px.
    Median { radius: usize },
    /// Sharpening: `x + amount · (x − blur(x))`.
    UnsharpMask { sigma: f64, amount: f64 },
    /// Band-pass: `blur(low) − blur(high)` (blob enhancement).
    DifferenceOfGaussians { low_sigma: f64, high_sigma: f64 },
    /// Gradient magnitude.
    Sobel {},
    /// Second derivative (edges, spots).
    Laplace {},
    /// `gain · x^gamma` (values in 0–1).
    Gamma { gamma: f64, gain: f64 },
    /// Maps `[min, max]` to `[0, 1]`, clipped.
    Rescale { min: f64, max: f64 },
    /// Contrast-limited adaptive histogram equalization.
    Clahe { clip_limit: f64, tiles: usize },
    /// Subtracts the rolling-ball background.
    RollingBall { radius: f64 },
    /// Total-variation denoising.
    TvDenoise { weight: f64, iterations: usize },
    /// Bright details smaller than the disk (background removal).
    WhiteTophat { radius: usize },
    /// Dilation − erosion (outlines).
    MorphGradient { radius: usize },
    /// Edges as a mask.
    Canny { sigma: f64 },
    /// Pixels above the Otsu threshold.
    OtsuThreshold { bins: usize },
    /// Pixels at or above `value`.
    Threshold { value: f64 },
    /// Mask opening (removes specks).
    Open { radius: usize },
    /// Mask closing (fills gaps).
    Close { radius: usize },
    Erode { radius: usize },
    Dilate { radius: usize },
    /// Drops objects smaller than `min_size` px.
    RemoveSmall { min_size: usize },
    /// Connected objects of a mask, numbered.
    Label {},
    /// Grows the marker labels over the relief image (splits touching objects).
    Watershed { compactness: f64 },
    /// The image where the mask is on, 0 elsewhere.
    MaskImage {},
    /// `a · A + b · B`.
    Combine { a: f64, b: f64 },
    /// One row per labelled object: area, shape, intensity.
    Measure {},
}

impl Op {
    /// The `op` tag (`gaussian_blur`, …).
    pub fn tag(&self) -> &'static str {
        match self {
            Op::GaussianBlur { .. } => "gaussian_blur",
            Op::Median { .. } => "median",
            Op::UnsharpMask { .. } => "unsharp_mask",
            Op::DifferenceOfGaussians { .. } => "difference_of_gaussians",
            Op::Sobel {} => "sobel",
            Op::Laplace {} => "laplace",
            Op::Gamma { .. } => "gamma",
            Op::Rescale { .. } => "rescale",
            Op::Clahe { .. } => "clahe",
            Op::RollingBall { .. } => "rolling_ball",
            Op::TvDenoise { .. } => "tv_denoise",
            Op::WhiteTophat { .. } => "white_tophat",
            Op::MorphGradient { .. } => "morph_gradient",
            Op::Canny { .. } => "canny",
            Op::OtsuThreshold { .. } => "otsu_threshold",
            Op::Threshold { .. } => "threshold",
            Op::Open { .. } => "open",
            Op::Close { .. } => "close",
            Op::Erode { .. } => "erode",
            Op::Dilate { .. } => "dilate",
            Op::RemoveSmall { .. } => "remove_small",
            Op::Label {} => "label",
            Op::Watershed { .. } => "watershed",
            Op::MaskImage {} => "mask_image",
            Op::Combine { .. } => "combine",
            Op::Measure {} => "measure",
        }
    }

    /// Its name for people.
    pub fn title(&self) -> &'static str {
        match self {
            Op::GaussianBlur { .. } => "Gaussian blur",
            Op::Median { .. } => "Median",
            Op::UnsharpMask { .. } => "Unsharp mask",
            Op::DifferenceOfGaussians { .. } => "Difference of Gaussians",
            Op::Sobel {} => "Sobel",
            Op::Laplace {} => "Laplace",
            Op::Gamma { .. } => "Gamma",
            Op::Rescale { .. } => "Rescale",
            Op::Clahe { .. } => "CLAHE",
            Op::RollingBall { .. } => "Rolling ball",
            Op::TvDenoise { .. } => "TV denoise",
            Op::WhiteTophat { .. } => "White top-hat",
            Op::MorphGradient { .. } => "Morph. gradient",
            Op::Canny { .. } => "Canny edges",
            Op::OtsuThreshold { .. } => "Otsu threshold",
            Op::Threshold { .. } => "Threshold",
            Op::Open { .. } => "Open",
            Op::Close { .. } => "Close",
            Op::Erode { .. } => "Erode",
            Op::Dilate { .. } => "Dilate",
            Op::RemoveSmall { .. } => "Remove small",
            Op::Label {} => "Label objects",
            Op::Watershed { .. } => "Watershed",
            Op::MaskImage {} => "Mask image",
            Op::Combine { .. } => "Combine",
            Op::Measure {} => "Measure",
        }
    }

    /// Its family (menus, colours).
    pub fn family(&self) -> &'static str {
        match self {
            Op::GaussianBlur { .. } | Op::Median { .. } | Op::UnsharpMask { .. } | Op::DifferenceOfGaussians { .. } | Op::Sobel {} | Op::Laplace {} => "Filter",
            Op::Gamma { .. } | Op::Rescale { .. } | Op::Clahe { .. } | Op::Combine { .. } | Op::MaskImage {} => "Intensity",
            Op::RollingBall { .. } | Op::TvDenoise { .. } | Op::WhiteTophat { .. } | Op::MorphGradient { .. } => "Restore",
            Op::Canny { .. } | Op::OtsuThreshold { .. } | Op::Threshold { .. } => "Threshold",
            Op::Open { .. } | Op::Close { .. } | Op::Erode { .. } | Op::Dilate { .. } | Op::RemoveSmall { .. } => "Mask",
            Op::Label {} | Op::Watershed { .. } => "Segment",
            Op::Measure {} => "Measure",
        }
    }

    /// Input types, in port order, with a name for each port.
    pub fn inputs(&self) -> Vec<(PortType, &'static str)> {
        use PortType::*;
        match self {
            Op::Open { .. } | Op::Close { .. } | Op::Erode { .. } | Op::Dilate { .. } | Op::RemoveSmall { .. } | Op::Label {} => vec![(Mask, "mask")],
            Op::Watershed { .. } => vec![(Image, "relief"), (Labels, "markers")],
            Op::MaskImage {} => vec![(Image, "image"), (Mask, "mask")],
            Op::Combine { .. } => vec![(Image, "A"), (Image, "B")],
            Op::Measure {} => vec![(Labels, "labels"), (Image, "intensity")],
            _ => vec![(Image, "image")],
        }
    }

    pub fn output(&self) -> PortType {
        use PortType::*;
        match self {
            Op::Canny { .. } | Op::OtsuThreshold { .. } | Op::Threshold { .. } | Op::Open { .. } | Op::Close { .. } | Op::Erode { .. } | Op::Dilate { .. } | Op::RemoveSmall { .. } => Mask,
            Op::Label {} | Op::Watershed { .. } => Labels,
            Op::Measure {} => Table,
            _ => Image,
        }
    }

    /// Pixels of context read around each output pixel (at the level it runs on). Ops that look
    /// at the whole input (thresholds, CLAHE, labelling) give their usual margin; their result on a
    /// region can differ from the whole image's.
    pub fn halo(&self) -> usize {
        let s = |sigma: f64| (4.0 * sigma.max(0.0)).ceil() as usize;
        match self {
            Op::GaussianBlur { sigma } | Op::UnsharpMask { sigma, .. } => s(*sigma),
            Op::DifferenceOfGaussians { low_sigma, high_sigma } => s(low_sigma.max(*high_sigma)),
            Op::Canny { sigma } => s(*sigma) + 2,
            Op::Median { radius } | Op::WhiteTophat { radius } | Op::MorphGradient { radius } | Op::Erode { radius } | Op::Dilate { radius } => *radius,
            Op::Open { radius } | Op::Close { radius } => 2 * radius,
            Op::Sobel {} | Op::Laplace {} => 1,
            Op::RollingBall { radius } => (2.0 * radius).ceil() as usize,
            Op::TvDenoise { iterations, .. } => (*iterations).min(32),
            Op::Clahe { .. } => 32,
            _ => 0,
        }
    }

    /// The op for a level whose pixels are `ds` level-0 pixels: sizes (sigmas, radii, areas) are
    /// given in level-0 px, so a result looks the same at every pyramid level.
    pub fn at_downsample(&self, ds: f64) -> Op {
        let ds = ds.max(1e-9);
        let s = |v: f64| (v / ds).max(0.3);
        let r = |v: usize| ((v as f64 / ds).round() as usize).max(1);
        match self.clone() {
            Op::GaussianBlur { sigma } => Op::GaussianBlur { sigma: s(sigma) },
            Op::Median { radius } => Op::Median { radius: r(radius) },
            Op::UnsharpMask { sigma, amount } => Op::UnsharpMask { sigma: s(sigma), amount },
            Op::DifferenceOfGaussians { low_sigma, high_sigma } => {
                let (lo, hi) = (s(low_sigma), s(high_sigma));
                Op::DifferenceOfGaussians { low_sigma: lo, high_sigma: hi.max(lo * 1.01) }
            }
            Op::RollingBall { radius } => Op::RollingBall { radius: (radius / ds).max(1.0) },
            Op::WhiteTophat { radius } => Op::WhiteTophat { radius: r(radius) },
            Op::MorphGradient { radius } => Op::MorphGradient { radius: r(radius) },
            Op::Canny { sigma } => Op::Canny { sigma: s(sigma) },
            Op::Open { radius } => Op::Open { radius: r(radius) },
            Op::Close { radius } => Op::Close { radius: r(radius) },
            Op::Erode { radius } => Op::Erode { radius: r(radius) },
            Op::Dilate { radius } => Op::Dilate { radius: r(radius) },
            Op::RemoveSmall { min_size } => Op::RemoveSmall { min_size: ((min_size as f64) / (ds * ds)).round() as usize },
            other => other,
        }
    }

    /// Whether its result depends on statistics of its whole input (so a region of the image
    /// needs the value computed on the whole image).
    pub fn is_global(&self) -> bool {
        matches!(self, Op::OtsuThreshold { .. })
    }

    /// Checks the parameters (positive sizes, finite numbers).
    pub fn validate(&self) -> Result<(), String> {
        let pos = |name: &str, v: f64| if v.is_finite() && v > 0.0 { Ok(()) } else { Err(format!("{name} must be a positive number")) };
        let fin = |name: &str, v: f64| if v.is_finite() { Ok(()) } else { Err(format!("{name} must be a number")) };
        match self {
            Op::GaussianBlur { sigma } | Op::Canny { sigma } => pos("sigma", *sigma),
            Op::UnsharpMask { sigma, amount } => pos("sigma", *sigma).and(fin("amount", *amount)),
            Op::DifferenceOfGaussians { low_sigma, high_sigma } => {
                pos("low sigma", *low_sigma)?;
                pos("high sigma", *high_sigma)?;
                if high_sigma <= low_sigma { Err("high sigma must be larger than low sigma".into()) } else { Ok(()) }
            }
            Op::Gamma { gamma, gain } => pos("gamma", *gamma).and(pos("gain", *gain)),
            Op::Rescale { min, max } => {
                if min.is_finite() && max.is_finite() && max > min { Ok(()) } else { Err("max must be larger than min".into()) }
            }
            Op::Clahe { clip_limit, tiles } => {
                pos("clip limit", *clip_limit)?;
                if *tiles >= 2 { Ok(()) } else { Err("tiles must be at least 2".into()) }
            }
            Op::RollingBall { radius } => pos("radius", *radius),
            Op::TvDenoise { weight, iterations } => {
                pos("weight", *weight)?;
                if *iterations >= 1 { Ok(()) } else { Err("iterations must be at least 1".into()) }
            }
            Op::Median { radius } | Op::WhiteTophat { radius } | Op::MorphGradient { radius } | Op::Open { radius } | Op::Close { radius } | Op::Erode { radius } | Op::Dilate { radius } => {
                if *radius >= 1 && *radius <= 64 { Ok(()) } else { Err("radius must be 1–64 px".into()) }
            }
            Op::OtsuThreshold { bins } => if (16..=4096).contains(bins) { Ok(()) } else { Err("bins must be 16–4096".into()) },
            Op::Threshold { value } => fin("value", *value),
            Op::Watershed { compactness } => if compactness.is_finite() && *compactness >= 0.0 { Ok(()) } else { Err("compactness must be ≥ 0".into()) },
            Op::Combine { a, b } => fin("a", *a).and(fin("b", *b)),
            Op::RemoveSmall { .. } | Op::Sobel {} | Op::Laplace {} | Op::Label {} | Op::MaskImage {} | Op::Measure {} => Ok(()),
        }
    }

    /// Runs the op on its inputs (in port order, already type-checked by the caller).
    pub fn run(&self, client: &Client, inputs: &[Value]) -> Result<Value, String> {
        self.validate()?;
        let e = |err: lumen_ops::OpsError| err.to_string();
        let image = |i: usize| match inputs.get(i) {
            Some(Value::Image(a)) => Ok(a),
            other => Err(format!("{}: input {} must be an image, got {}", self.title(), i + 1, other.map_or("nothing", |v| v.port().name()))),
        };
        let mask = |i: usize| match inputs.get(i) {
            Some(Value::Mask(a)) => Ok(a),
            other => Err(format!("{}: input {} must be a mask, got {}", self.title(), i + 1, other.map_or("nothing", |v| v.port().name()))),
        };
        let labels = |i: usize| match inputs.get(i) {
            Some(Value::Labels(a)) => Ok(a),
            other => Err(format!("{}: input {} must be labels, got {}", self.title(), i + 1, other.map_or("nothing", |v| v.port().name()))),
        };
        let disk = |r: usize| StructuringElement::disk(r);
        Ok(match self {
            Op::GaussianBlur { sigma } => Value::Image(gaussian_blur(client, image(0)?, &[*sigma, *sigma], &GaussianBlur::default()).map_err(e)?),
            Op::Median { radius } => Value::Image(median_filter(client, image(0)?, &disk(*radius), &OrderFilter::default()).map_err(e)?),
            Op::UnsharpMask { sigma, amount } => Value::Image(unsharp_mask(client, image(0)?, &[*sigma, *sigma], *amount, &GaussianBlur::default()).map_err(e)?),
            Op::DifferenceOfGaussians { low_sigma, high_sigma } => {
                Value::Image(difference_of_gaussians(client, image(0)?, &[*low_sigma, *low_sigma], &[*high_sigma, *high_sigma], &GaussianBlur::default()).map_err(e)?)
            }
            Op::Sobel {} => Value::Image(sobel(client, image(0)?, &GradientFilter::default()).map_err(e)?),
            Op::Laplace {} => Value::Image(laplace(client, image(0)?, lumen_ops::core::PadMode::Reflect, 0.0).map_err(e)?),
            Op::Gamma { gamma, gain } => Value::Image(adjust_gamma(client, image(0)?, *gamma, *gain).map_err(e)?),
            Op::Rescale { min, max } => {
                let a = image(0)?;
                let scaled = DeviceArray::<f32>::empty(client, a.shape().clone()).map_err(|err| err.to_string())?;
                let scale = 1.0 / (max - min);
                elementwise::scale_offset_into(client, a, &scaled, scale, -min * scale).map_err(e)?;
                let out = DeviceArray::<f32>::empty(client, a.shape().clone()).map_err(|err| err.to_string())?;
                elementwise::clip_into(client, &scaled, &out, 0.0, 1.0).map_err(e)?;
                Value::Image(out)
            }
            Op::Clahe { clip_limit, tiles } => Value::Image(clahe(client, image(0)?, ClaheSettings { grid_size: (*tiles, *tiles), clip_limit: *clip_limit, num_bins: 256 }).map_err(e)?),
            Op::RollingBall { radius } => {
                let a = image(0)?;
                let bg = rolling_ball(client, a, &RollingBall { radius: *radius, ..RollingBall::default() }).map_err(e)?;
                let out = DeviceArray::<f32>::empty(client, a.shape().clone()).map_err(|err| err.to_string())?;
                elementwise::linear_combination_into(client, a, &bg, &out, 1.0, -1.0, 0.0).map_err(e)?;
                Value::Image(out)
            }
            Op::TvDenoise { weight, iterations } => Value::Image(tv_chambolle(client, image(0)?, &TvChambolle { weight: *weight, max_num_iter: *iterations, ..TvChambolle::default() }).map_err(e)?),
            Op::WhiteTophat { radius } => Value::Image(white_tophat(client, image(0)?, &disk(*radius), &GrayMorphology::default()).map_err(e)?),
            Op::MorphGradient { radius } => Value::Image(morph_gradient(client, image(0)?, &disk(*radius), &GrayMorphology::default()).map_err(e)?),
            Op::Canny { sigma } => Value::Mask(canny(client, image(0)?, &Canny { sigma: *sigma, ..Canny::default() }).map_err(e)?),
            Op::OtsuThreshold { bins } => {
                let a = image(0)?;
                let t = otsu(client, a, *bins).map_err(e)?;
                Value::Mask(threshold(client, a, t)?)
            }
            Op::Threshold { value } => Value::Mask(threshold(client, image(0)?, *value)?),
            Op::Open { radius } => Value::Mask(binary_open(client, mask(0)?, &disk(*radius), &BinaryMorphology::default()).map_err(e)?),
            Op::Close { radius } => Value::Mask(binary_close(client, mask(0)?, &disk(*radius), &BinaryMorphology::default()).map_err(e)?),
            Op::Erode { radius } => Value::Mask(binary_erode(client, mask(0)?, &disk(*radius), &BinaryMorphology::default()).map_err(e)?),
            Op::Dilate { radius } => Value::Mask(binary_dilate(client, mask(0)?, &disk(*radius), &BinaryMorphology::default()).map_err(e)?),
            Op::RemoveSmall { min_size } => Value::Mask(remove_small_objects(client, mask(0)?, *min_size, None).map_err(e)?),
            Op::Label {} => Value::Labels(label_components(client, mask(0)?, None).map_err(e)?.labels),
            Op::Watershed { compactness } => {
                let opts = WatershedOptions { compactness: *compactness, ..WatershedOptions::default() };
                Value::Labels(watershed(client, image(0)?, labels(1)?, None, None, &opts).map_err(e)?)
            }
            Op::MaskImage {} => {
                // No device product kernel yet: the region is small, multiply on the host
                let (a, m) = (image(0)?, mask(1)?);
                if a.dims() != m.dims() {
                    return Err("Mask image: the image and the mask differ in size".into());
                }
                let (va, vm) = (a.to_host(client), m.to_host(client));
                let out: Vec<f32> = va.iter().zip(vm.iter()).map(|(v, &k)| if k != 0 { *v } else { 0.0 }).collect();
                Value::Image(DeviceArray::from_host(client, a.shape().clone(), out).map_err(|err| err.to_string())?)
            }
            Op::Combine { a, b } => {
                let (x, y) = (image(0)?, image(1)?);
                if x.dims() != y.dims() {
                    return Err("Combine: A and B differ in size".into());
                }
                let out = DeviceArray::<f32>::empty(client, x.shape().clone()).map_err(|err| err.to_string())?;
                elementwise::linear_combination_into(client, x, y, &out, *a, *b, 0.0).map_err(e)?;
                Value::Image(out)
            }
            Op::Measure {} => Value::Table(region_features(client, labels(0)?, Some(image(1)?)).map_err(e)?),
        })
    }
}

fn threshold(client: &Client, a: &DeviceArray<f32>, t: f64) -> Result<DeviceArray<u32>, String> {
    let m = DeviceArray::<u32>::empty(client, a.shape().clone()).map_err(|e| e.to_string())?;
    elementwise::threshold_scalar_into(client, a, t, true, &m).map_err(|e| e.to_string())?;
    Ok(m)
}

/// Every op with its default parameters, in menu order.
pub fn catalog() -> Vec<Op> {
    vec![
        Op::GaussianBlur { sigma: 2.0 },
        Op::Median { radius: 2 },
        Op::UnsharpMask { sigma: 2.0, amount: 1.0 },
        Op::DifferenceOfGaussians { low_sigma: 1.0, high_sigma: 4.0 },
        Op::Sobel {},
        Op::Laplace {},
        Op::Gamma { gamma: 0.7, gain: 1.0 },
        Op::Rescale { min: 0.0, max: 0.25 },
        Op::Clahe { clip_limit: 0.01, tiles: 8 },
        Op::Combine { a: 1.0, b: -1.0 },
        Op::MaskImage {},
        Op::RollingBall { radius: 25.0 },
        Op::TvDenoise { weight: 0.1, iterations: 50 },
        Op::WhiteTophat { radius: 8 },
        Op::MorphGradient { radius: 2 },
        Op::OtsuThreshold { bins: 256 },
        Op::Threshold { value: 0.1 },
        Op::Canny { sigma: 1.5 },
        Op::Open { radius: 2 },
        Op::Close { radius: 2 },
        Op::Erode { radius: 1 },
        Op::Dilate { radius: 1 },
        Op::RemoveSmall { min_size: 30 },
        Op::Label {},
        Op::Watershed { compactness: 0.0 },
        Op::Measure {},
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_round_trip_as_json_and_have_valid_defaults() {
        for op in catalog() {
            let text = serde_json::to_string(&op).unwrap();
            assert!(text.contains(&format!("\"op\":\"{}\"", op.tag())), "{text}");
            assert_eq!(serde_json::from_str::<Op>(&text).unwrap(), op);
            assert!(op.validate().is_ok(), "{}", op.title());
            assert!(!op.inputs().is_empty());
        }
        assert!(serde_json::from_str::<Op>(r#"{"op":"gaussian_blur","sigma":1,"sigmaa":2}"#).is_err(), "unknown fields fail");
    }

    #[test]
    fn ports_are_typed() {
        assert_eq!(Op::Label {}.inputs()[0].0, PortType::Mask);
        assert_eq!(Op::Label {}.output(), PortType::Labels);
        assert_eq!(Op::Measure {}.inputs().len(), 2);
        assert_eq!(Op::OtsuThreshold { bins: 256 }.output(), PortType::Mask);
    }
}
