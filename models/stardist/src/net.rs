//! StarDist2D's network as a Burn module: the layers of the published H&E model (a U-Net of depth
//! 3 from 32 filters, two 3 × 3 convolutions per level, ReLU, after a first 2 × 2 pooling for its
//! grid of 2 × 2), a 128-filter 3 × 3 `features` convolution, and two 1 × 1 heads: `prob`
//! (sigmoid, the object probability) and `dist` (32 distances from the pixel to the object's edge
//! along 32 rays). Both come out on the grid (half the input in each axis).
//!
//! [`StarDistModule`] is a `#[derive(Module)]` of `burn::nn` layers whose fields are named as the
//! published Keras layers, so the safetensors written at install ([`crate::convert`]) load into it
//! by name through Burn's `SafetensorsStore`. Burn 0.22's tensors carry their device, which picks
//! the backend (CUDA, wgpu, the CPU): one module serves every runtime.

use std::path::Path;

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::PaddingConfig2d;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::ops::PadMode;
use burn::tensor::{Device, FloatDType, Tensor};

/// A convolution of the published model: its Keras name, kernel side, filters in and out.
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    pub name: &'static str,
    pub kernel: usize,
    pub input: usize,
    pub output: usize,
}

const fn l(name: &'static str, kernel: usize, input: usize, output: usize) -> Layer {
    Layer { name, kernel, input, output }
}

/// The convolutions, in the order the network runs them.
pub const LAYERS: [Layer; 19] = [
    l("conv2d_1", 3, 3, 32),
    l("conv2d_2", 3, 32, 32),
    l("down_level_0_no_0", 3, 32, 32),
    l("down_level_0_no_1", 3, 32, 32),
    l("down_level_1_no_0", 3, 32, 64),
    l("down_level_1_no_1", 3, 64, 64),
    l("down_level_2_no_0", 3, 64, 128),
    l("down_level_2_no_1", 3, 128, 128),
    l("middle_0", 3, 128, 256),
    l("middle_2", 3, 256, 128),
    l("up_level_2_no_0", 3, 256, 128),
    l("up_level_2_no_2", 3, 128, 64),
    l("up_level_1_no_0", 3, 128, 64),
    l("up_level_1_no_2", 3, 64, 32),
    l("up_level_0_no_0", 3, 64, 32),
    l("up_level_0_no_2", 3, 32, 32),
    l("features", 3, 32, 128),
    l("prob", 1, 128, 1),
    l("dist", 1, 128, 32),
];

/// The layers of a model whose images have `channels` channels (only the first layer differs).
pub fn layers(channels: usize) -> Vec<Layer> {
    let mut l = LAYERS.to_vec();
    l[0].input = channels;
    l
}

/// How much the input shrinks before the deepest level: sizes must be multiples of it.
pub const DIVISOR: usize = 16;
/// The output grid (input pixels per output pixel, each axis).
pub const GRID: usize = 2;
pub const N_RAYS: usize = 32;

/// The network: one field per published layer, named as in the Keras model.
#[derive(Module, Debug)]
pub struct StarDistModule {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
    down_level_0_no_0: Conv2d,
    down_level_0_no_1: Conv2d,
    down_level_1_no_0: Conv2d,
    down_level_1_no_1: Conv2d,
    down_level_2_no_0: Conv2d,
    down_level_2_no_1: Conv2d,
    middle_0: Conv2d,
    middle_2: Conv2d,
    up_level_2_no_0: Conv2d,
    up_level_2_no_2: Conv2d,
    up_level_1_no_0: Conv2d,
    up_level_1_no_2: Conv2d,
    up_level_0_no_0: Conv2d,
    up_level_0_no_2: Conv2d,
    features: Conv2d,
    prob: Conv2d,
    dist: Conv2d,
    pool: MaxPool2d,
    up: Interpolate2d,
    /// The 3 × 3 layers of 64 channels and more run as matrix products ([`conv_gemm`]).
    #[module(skip)]
    gemm: bool,
}

/// The width from which 3 × 3 convolutions run as matrix products ([`conv_gemm`]).
const GEMM_FROM_CHANNELS: usize = 64;

/// Set when a matrix product could not launch on this device: every layer runs Burn's own
/// convolution from then on ([`disable_gemm`]).
static GEMM_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns the matrix-product convolutions off for this process; whether they were on.
pub fn disable_gemm() -> bool {
    !GEMM_OFF.swap(true, std::sync::atomic::Ordering::Relaxed)
}

/// A 3 × 3 convolution as one matrix product: the input zero-padded, its nine shifted copies
/// stacked on the channel axis (`[n, 9c, h·w]`, offset-major), times the weights reordered to
/// match (`[o, 9c]`). Burn's tensor-core convolution asks more shared memory than a Turing GPU has
/// (64 KB per block) for layers of 64 channels and more and falls back to its direct kernel, about
/// 15× slower; its matrix product tunes and runs on the tensor cores there.
fn conv_gemm(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let [n, c, h, w] = x.dims();
    let [o, _, kh, kw] = conv.weight.dims();
    let (ph, pw) = (kh / 2, kw / 2);
    let xp = x.pad((pw, pw, ph, ph), PadMode::Constant(0.0));
    let shifted: Vec<Tensor<4>> = (0..kh).flat_map(|dy| (0..kw).map(move |dx| (dy, dx))).map(|(dy, dx)| xp.clone().slice([0..n, 0..c, dy..dy + h, dx..dx + w])).collect();
    let cols = Tensor::cat(shifted, 1).reshape([n, kh * kw * c, h * w]);
    let weights = conv.weight.val().permute([0, 2, 3, 1]).reshape([1, o, kh * kw * c]);
    let mut y = weights.matmul(cols);
    if let Some(b) = &conv.bias {
        y = y + b.val().reshape([1, o, 1]);
    }
    y.reshape([n, o, h, w])
}

/// Casts every float parameter of a module to `dtype`.
struct Cast(FloatDType);

impl ModuleMapper for Cast {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dtype = self.0;
        param.map(|t| t.cast(dtype))
    }
}

impl StarDistModule {
    /// The layers for `channels`-channel images, before their weights are loaded.
    pub fn new(channels: usize, device: &Device) -> Self {
        let c = |i: usize| {
            let l = layers(channels)[i];
            let p = l.kernel / 2;
            Conv2dConfig::new([l.input, l.output], [l.kernel, l.kernel]).with_padding(PaddingConfig2d::Explicit(p, p, p, p)).init(device)
        };
        Self {
            conv2d_1: c(0),
            conv2d_2: c(1),
            down_level_0_no_0: c(2),
            down_level_0_no_1: c(3),
            down_level_1_no_0: c(4),
            down_level_1_no_1: c(5),
            down_level_2_no_0: c(6),
            down_level_2_no_1: c(7),
            middle_0: c(8),
            middle_2: c(9),
            up_level_2_no_0: c(10),
            up_level_2_no_2: c(11),
            up_level_1_no_0: c(12),
            up_level_1_no_2: c(13),
            up_level_0_no_0: c(14),
            up_level_0_no_2: c(15),
            features: c(16),
            prob: c(17),
            dist: c(18),
            gemm: false,
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            up: Interpolate2dConfig::new().with_scale_factor(Some([2.0, 2.0])).with_mode(InterpolateMode::Nearest).init(),
        }
    }

    /// `x`: `[n, channels, h, w]` (normalized; `h`, `w` multiples of [`DIVISOR`]) → probability
    /// `[n, 1, h/2, w/2]` and distances `[n, 32, h/2, w/2]` (input pixels).
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let gemm = self.gemm && !GEMM_OFF.load(std::sync::atomic::Ordering::Relaxed);
        let run = move |conv: &Conv2d, x: Tensor<4>| {
            let [o, i, kh, _] = conv.weight.dims();
            if gemm && kh > 1 && o.max(i) >= GEMM_FROM_CHANNELS { conv_gemm(conv, x) } else { conv.forward(x) }
        };
        let c = |conv: &Conv2d, x: Tensor<4>| relu(run(conv, x));
        // To the grid
        let x = self.pool.forward(c(&self.conv2d_2, c(&self.conv2d_1, x)));
        // Down
        let s0 = c(&self.down_level_0_no_1, c(&self.down_level_0_no_0, x));
        let s1 = c(&self.down_level_1_no_1, c(&self.down_level_1_no_0, self.pool.forward(s0.clone())));
        let s2 = c(&self.down_level_2_no_1, c(&self.down_level_2_no_0, self.pool.forward(s1.clone())));
        let m = c(&self.middle_2, c(&self.middle_0, self.pool.forward(s2.clone())));
        // Up: upsampled, then the level's skip
        let u2 = c(&self.up_level_2_no_2, c(&self.up_level_2_no_0, Tensor::cat(vec![self.up.forward(m), s2], 1)));
        let u1 = c(&self.up_level_1_no_2, c(&self.up_level_1_no_0, Tensor::cat(vec![self.up.forward(u2), s1], 1)));
        let u0 = c(&self.up_level_0_no_2, c(&self.up_level_0_no_0, Tensor::cat(vec![self.up.forward(u1), s0], 1)));
        let f = c(&self.features, u0);
        (sigmoid(run(&self.prob, f.clone())), run(&self.dist, f))
    }
}

/// The precision the network runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    /// Half precision: weights, activations and the data to and from the device (half the bytes
    /// moved), and the GPU's tensor-core convolutions.
    F16,
}

impl Precision {
    /// `LUCEM_STARDIST_PRECISION` (`f32`, `f16`), else half precision on a GPU and single
    /// precision on the CPU runtime (`LUCEM_RUNTIME=cpu`).
    pub fn from_env() -> Self {
        match std::env::var("LUCEM_STARDIST_PRECISION").unwrap_or_default().to_lowercase().as_str() {
            "f32" => Precision::F32,
            "f16" => Precision::F16,
            _ if std::env::var("LUCEM_RUNTIME").is_ok_and(|r| r.eq_ignore_ascii_case("cpu")) => Precision::F32,
            _ => Precision::F16,
        }
    }
}

/// The network on its device, for images of `channels` channels.
pub struct StarDistNet {
    pub module: StarDistModule,
    pub device: Device,
    pub channels: usize,
    pub precision: Precision,
}

impl StarDistNet {
    /// Loads the safetensors written at install into the module on `device`: every layer must be
    /// found and every tensor of the file used.
    pub fn load(path: &Path, channels: usize, device: Device) -> Result<Self, String> {
        let mut module = StarDistModule::new(channels, &device);
        let mut store = SafetensorsStore::from_file(path);
        let r = module.load_from(&mut store).map_err(|e| format!("{}: {e}", path.display()))?;
        if !r.errors.is_empty() || !r.missing.is_empty() || !r.unused.is_empty() {
            return Err(format!("{}: errors {:?}, missing {:?}, unused {:?}", path.display(), r.errors, r.missing, r.unused));
        }
        let precision = Precision::from_env();
        // The wide 3 × 3 layers as matrix products on a GPU in half precision (`conv_gemm`);
        // `LUCEM_STARDIST_CONV=gemm|direct` chooses
        module.gemm = match std::env::var("LUCEM_STARDIST_CONV").unwrap_or_default().to_lowercase().as_str() {
            "gemm" => true,
            "direct" => false,
            _ => precision == Precision::F16,
        };
        if precision == Precision::F16 {
            module = module.map(&mut Cast(FloatDType::F16));
        }
        Ok(Self { module, device, channels, precision })
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        self.module.forward(x)
    }
}
