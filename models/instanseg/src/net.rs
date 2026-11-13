//! InstanSeg's networks as Burn modules, mirroring the PyTorch classes of the published models:
//!
//! - [`UNet`] (`InstanSeg_UNet`): [`EncoderBlock`]s (a 1 × 1 projection plus two 3 × 3
//!   convolutions, then two more as a residual; 2 × 2 max pooling before every block but the
//!   first), [`DecoderBlock`]s (nearest ×2 upsampling, the same residual shape with the skip
//!   connection's 1 × 1 projection added in) and, per [`Decoder`], 1 × 1 output blocks
//!   concatenated: the pixel embedding offsets (2), their sigma (2) and the seed map (1).
//! - [`ProbabilityNet`]: the pixel classifier (4 → 5 → 5 → 1, ReLU between).
//!
//! Every convolution is a [`ConvNormAct`] (`conv_norm_act` upstream): `Conv2d`, `BatchNorm`
//! (Burn takes its inference path, the running statistics, on a device without autodiff), then
//! ReLU except for the output blocks. Block and channel counts come from the weights
//! ([`UNetConfig::from_shapes`]), and the weights load by name through Burn's
//! `SafetensorsStore` with its PyTorch adapter ([`load`]).

use std::collections::BTreeMap;
use std::path::Path;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Tensor};

/// How much the input shrinks at the deepest level: sizes must be multiples of it (InstanSeg
/// pads to 16).
pub const DIVISOR: usize = 16;

/// BatchNorm's epsilon in the published models.
const BN_EPS: f64 = 1e-5;

/// A convolution, its BatchNorm, and ReLU unless it is an output block.
#[derive(Module, Debug)]
pub struct ConvNormAct {
    conv: Conv2d,
    norm: BatchNorm,
    #[module(skip)]
    relu: bool,
}

impl ConvNormAct {
    fn new(input: usize, output: usize, kernel: usize, relu: bool, device: &Device) -> Self {
        let p = kernel / 2;
        Self {
            conv: Conv2dConfig::new([input, output], [kernel, kernel]).with_padding(PaddingConfig2d::Explicit(p, p, p, p)).init(device),
            norm: BatchNormConfig::new(output).with_epsilon(BN_EPS).init(device),
            relu,
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = self.norm.forward(self.conv.forward(x));
        if self.relu { relu(y) } else { y }
    }
}

fn apply(c: &Option<ConvNormAct>, x: Tensor<4>) -> Tensor<4> {
    match c {
        Some(c) => c.forward(x),
        None => x,
    }
}

#[derive(Module, Debug)]
pub struct EncoderBlock {
    #[module(skip)]
    pool: bool,
    maxpool: MaxPool2d,
    conv0: ConvNormAct,
    conv1: ConvNormAct,
    /// `None` in InstanSeg's shallow blocks (an identity there).
    conv2: Option<ConvNormAct>,
    conv3: Option<ConvNormAct>,
    conv4: ConvNormAct,
}

impl EncoderBlock {
    fn new(input: usize, output: usize, pool: bool, shallow: bool, device: &Device) -> Self {
        let c = |i, k| ConvNormAct::new(i, output, k, true, device);
        Self {
            pool,
            maxpool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv0: c(input, 1),
            conv1: c(input, 3),
            conv2: (!shallow).then(|| c(output, 3)),
            conv3: (!shallow).then(|| c(output, 3)),
            conv4: c(output, 3),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = if self.pool { self.maxpool.forward(x) } else { x };
        let proj = self.conv0.forward(x.clone());
        let x = proj + apply(&self.conv2, self.conv1.forward(x));
        x.clone() + self.conv4.forward(apply(&self.conv3, x))
    }
}

#[derive(Module, Debug)]
pub struct DecoderBlock {
    up: Interpolate2d,
    conv0: ConvNormAct,
    conv_skip: ConvNormAct,
    conv1: ConvNormAct,
    conv2: ConvNormAct,
    conv3: Option<ConvNormAct>,
    conv4: ConvNormAct,
}

impl DecoderBlock {
    fn new(input: usize, skip: usize, output: usize, shallow: bool, device: &Device) -> Self {
        let c = |i, k| ConvNormAct::new(i, output, k, true, device);
        Self {
            up: Interpolate2dConfig::new().with_scale_factor(Some([2.0, 2.0])).with_mode(InterpolateMode::Nearest).init(),
            conv0: c(input, 1),
            conv_skip: ConvNormAct::new(skip, output, 1, true, device),
            conv1: c(input, 3),
            conv2: c(output, 3),
            conv3: (!shallow).then(|| c(output, 3)),
            conv4: c(output, 3),
        }
    }

    fn forward(&self, x: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        let x = self.up.forward(x);
        let proj = self.conv0.forward(x.clone());
        let x = self.conv1.forward(x);
        let x = proj + self.conv2.forward(x + self.conv_skip.forward(skip));
        x.clone() + self.conv4.forward(apply(&self.conv3, x))
    }
}

#[derive(Module, Debug)]
pub struct Decoder {
    decoder: Vec<DecoderBlock>,
    final_block: Vec<ConvNormAct>,
}

impl Decoder {
    fn forward(&self, x: Tensor<4>, skips: &[Tensor<4>]) -> Tensor<4> {
        let mut x = x;
        for (block, skip) in self.decoder.iter().zip(skips.iter().rev()) {
            x = block.forward(x, skip.clone());
        }
        Tensor::cat(self.final_block.iter().map(|b| b.forward(x.clone())).collect(), 1)
    }
}

/// `InstanSeg_UNet`.
#[derive(Module, Debug)]
pub struct UNet {
    encoder: Vec<EncoderBlock>,
    decoders: Vec<Decoder>,
}

impl UNet {
    /// `x`: `[n, channels, h, w]` (normalized; `h`, `w` multiples of [`DIVISOR`]) →
    /// `[n, outputs, h, w]`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut skips = Vec::new();
        let mut x = x;
        for (n, block) in self.encoder.iter().enumerate() {
            x = block.forward(x);
            if n + 1 < self.encoder.len() {
                skips.push(x.clone());
            }
        }
        Tensor::cat(self.decoders.iter().map(|d| d.forward(x.clone(), &skips)).collect(), 1)
    }
}

/// The pixel classifier.
#[derive(Module, Debug)]
pub struct ProbabilityNet {
    pub fc1: Linear,
    pub fc2: Linear,
    pub fc3: Linear,
}

/// What the published model holds: the network (`fcn`) and its pixel classifier.
#[derive(Module, Debug)]
pub struct InstanSegModel {
    pub fcn: UNet,
    pub pixel_classifier: ProbabilityNet,
}

/// The shape of a model, read from its weights.
#[derive(Debug, Clone, PartialEq)]
pub struct UNetConfig {
    pub channels: usize,
    /// Encoder widths, shallow first (`[32, 64, 128, 256]`).
    pub layers: Vec<usize>,
    /// Per decoder, its output blocks' widths (`[[2, 2, 1]]`).
    pub heads: Vec<Vec<usize>>,
    /// Shallow blocks (no `conv2`/`conv3`).
    pub shallow: bool,
    /// The pixel classifier's widths (`[4, 5, 5, 1]`).
    pub classifier: Vec<usize>,
}

impl UNetConfig {
    /// From the weights' names and shapes (PyTorch's, as in the archive).
    pub fn from_shapes(shapes: &BTreeMap<String, Vec<usize>>) -> Result<Self, String> {
        let out_of = |k: &str| shapes.get(k).map(|s| s[0]).ok_or_else(|| format!("{k} is missing"));
        let channels = shapes.get("fcn.encoder.0.conv0.0.weight").map(|s| s[1]).ok_or("fcn.encoder.0.conv0.0.weight is missing")?;
        let layers: Vec<usize> = (0..).map_while(|i| out_of(&format!("fcn.encoder.{i}.conv0.0.weight")).ok()).collect();
        if layers.len() < 2 {
            return Err(format!("{} encoder block(s) in the weights", layers.len()));
        }
        let heads: Vec<Vec<usize>> = (0..).map_while(|d| {
            let h: Vec<usize> = (0..).map_while(|k| out_of(&format!("fcn.decoders.{d}.final_block.{k}.0.weight")).ok()).collect();
            (!h.is_empty()).then_some(h)
        }).collect();
        if heads.is_empty() {
            return Err("no decoder in the weights".into());
        }
        let shallow = !shapes.contains_key("fcn.encoder.0.conv2.0.weight");
        let mut classifier = vec![shapes.get("pixel_classifier.fc1.weight").map(|s| s[1]).ok_or("pixel_classifier.fc1.weight is missing")?];
        classifier.extend((1..).map_while(|k| out_of(&format!("pixel_classifier.fc{k}.weight")).ok()));
        Ok(Self { channels, layers, heads, shallow, classifier })
    }

    /// Channels the network gives.
    pub fn outputs(&self) -> usize {
        self.heads.iter().flatten().sum()
    }

    /// The module, before its weights are loaded.
    pub fn init(&self, device: &Device) -> InstanSegModel {
        let l = &self.layers;
        let encoder = (0..l.len()).map(|i| EncoderBlock::new(if i == 0 { self.channels } else { l[i - 1] }, l[i], i > 0, self.shallow, device)).collect();
        // Decoders walk back up: from the deepest width to the shallowest
        let up: Vec<usize> = l.iter().rev().copied().collect();
        let decoders = self
            .heads
            .iter()
            .map(|heads| Decoder {
                decoder: (0..up.len() - 1).map(|i| DecoderBlock::new(up[i], up[i + 1], up[i + 1], self.shallow, device)).collect(),
                final_block: heads.iter().map(|&o| ConvNormAct::new(up[up.len() - 1], o, 1, false, device)).collect(),
            })
            .collect();
        let c = &self.classifier;
        let lin = |k: usize| LinearConfig::new(c[k], c[k + 1]).init(device);
        InstanSegModel { fcn: UNet { encoder, decoders }, pixel_classifier: ProbabilityNet { fc1: lin(0), fc2: lin(1), fc3: lin(2) } }
    }
}

/// The safetensors written at install (the archive's tensors under PyTorch's names) loaded into
/// the module: `Sequential` indices become the block's fields (`….0.weight` → `….conv.weight`,
/// `….1.running_mean` → `….norm.running_mean`), and Burn's PyTorch adapter does the rest
/// (BatchNorm's `weight`/`bias` as `gamma`/`beta`, linear weights transposed). Every module tensor
/// must be found; the archive's extras (`num_batches_tracked`, `fcn.pixel_classifier`, a copy of
/// the classifier) are left out.
pub fn load(path: &Path, config: &UNetConfig, device: &Device) -> Result<InstanSegModel, String> {
    let mut model = config.init(device);
    let mut store = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"^(.*)\.0\.(weight|bias)$", "$1.conv.$2")
        .with_key_remapping(r"^(.*)\.1\.(weight|bias|running_mean|running_var)$", "$1.norm.$2");
    let r = model.load_from(&mut store).map_err(|e| format!("{}: {e}", path.display()))?;
    if !r.errors.is_empty() || !r.missing.is_empty() {
        return Err(format!("{}: errors {:?}, missing {:?}", path.display(), r.errors, r.missing));
    }
    if let Some(odd) = r.unused.iter().find(|k| !(k.ends_with("num_batches_tracked") || k.starts_with("fcn.pixel_classifier."))) {
        return Err(format!("{}: {odd} has no place in the network", path.display()));
    }
    Ok(model)
}
