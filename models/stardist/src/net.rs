//! StarDist2D's network on Burn: the layers of the published H&E model (a U-Net of depth 3 from
//! 32 filters, two 3 × 3 convolutions per level, ReLU, after a first 2 × 2 pooling for its grid of
//! 2 × 2), a 128-filter 3 × 3 `features` convolution, and two 1 × 1 heads: `prob` (sigmoid, the
//! object probability) and `dist` (32 distances from the pixel to the object's edge along 32
//! rays). Both come out on the grid (half the input in each axis).
//!
//! Burn 0.22's tensors carry their device, which picks the backend (CUDA, wgpu, the CPU): one
//! network serves every runtime. Fusion is on: no tensor crosses into lumen's own kernels.

use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::module::{conv2d, interpolate, max_pool2d};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, MaxPoolOptions};
use burn::tensor::{Device, Tensor, TensorData};

use crate::convert::LayerWeights;

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

struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
    padding: usize,
}

impl Conv {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        conv2d(x, self.weight.clone(), Some(self.bias.clone()), ConvOptions::new([1, 1], [self.padding, self.padding], [1, 1], 1))
    }
}

pub struct StarDistNet {
    convs: Vec<Conv>,
    pub device: Device,
    /// Channels its images have.
    pub channels: usize,
}

fn pool(x: Tensor<4>) -> Tensor<4> {
    max_pool2d(x, MaxPoolOptions::new([2, 2]).with_stride([2, 2]))
}

fn up(x: Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    interpolate(x, InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([h * 2, w * 2]))
}

impl StarDistNet {
    /// The network on `device` from weights in [`LAYERS`]' order, for `channels`-channel images.
    pub fn new(weights: Vec<LayerWeights>, channels: usize, device: Device) -> Result<Self, String> {
        let mut convs = Vec::new();
        for (layer, w) in layers(channels).iter().zip(weights) {
            if w.name != layer.name {
                return Err(format!("weights for {}, expected {}", w.name, layer.name));
            }
            let weight = Tensor::<4>::from_data(TensorData::new(w.weight, w.shape), &device);
            let bias = Tensor::<1>::from_data(TensorData::new(w.bias, [layer.output]), &device);
            convs.push(Conv { weight, bias, padding: layer.kernel / 2 });
        }
        if convs.len() != LAYERS.len() {
            return Err(format!("{} layers of weights, expected {}", convs.len(), LAYERS.len()));
        }
        Ok(Self { convs, device, channels })
    }

    /// `x`: `[n, channels, h, w]` (normalized; `h`, `w` multiples of [`DIVISOR`]) → probability
    /// `[n, 1, h/2, w/2]` and distances `[n, 32, h/2, w/2]` (input pixels).
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let c = |i: usize, x: Tensor<4>| relu(self.convs[i].forward(x));
        // To the grid
        let x = pool(c(1, c(0, x)));
        // Down
        let s0 = c(3, c(2, x));
        let s1 = c(5, c(4, pool(s0.clone())));
        let s2 = c(7, c(6, pool(s1.clone())));
        let m = c(9, c(8, pool(s2.clone())));
        // Up: upsampled, then the level's skip
        let u2 = c(11, c(10, Tensor::cat(vec![up(m), s2], 1)));
        let u1 = c(13, c(12, Tensor::cat(vec![up(u2), s1], 1)));
        let u0 = c(15, c(14, Tensor::cat(vec![up(u1), s0], 1)));
        let f = c(16, u0);
        let prob = sigmoid(self.convs[17].forward(f.clone()));
        let dist = self.convs[18].forward(f);
        (prob, dist)
    }
}
