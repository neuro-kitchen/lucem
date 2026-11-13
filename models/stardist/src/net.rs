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

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::PaddingConfig2d;
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::{Device, Tensor};

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
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            up: Interpolate2dConfig::new().with_scale_factor(Some([2.0, 2.0])).with_mode(InterpolateMode::Nearest).init(),
        }
    }

    /// `x`: `[n, channels, h, w]` (normalized; `h`, `w` multiples of [`DIVISOR`]) → probability
    /// `[n, 1, h/2, w/2]` and distances `[n, 32, h/2, w/2]` (input pixels).
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let c = |conv: &Conv2d, x: Tensor<4>| relu(conv.forward(x));
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
        (sigmoid(self.prob.forward(f.clone())), self.dist.forward(f))
    }
}

/// The network on its device, for images of `channels` channels.
pub struct StarDistNet {
    pub module: StarDistModule,
    pub device: Device,
    pub channels: usize,
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
        Ok(Self { module, device, channels })
    }

    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        self.module.forward(x)
    }
}
