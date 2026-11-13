//! InstanSeg's U-Net on Burn (`InstanSeg_UNet` of the published models): encoder blocks (a 1 × 1
//! projection plus two 3 × 3 convolutions, then two more as a residual; 2 × 2 max pooling before
//! every block but the first), decoder blocks (nearest ×2 upsampling, the same residual shape with
//! the skip connection's 1 × 1 projection added in) and, per decoder, 1 × 1 output blocks
//! concatenated: the pixel embedding offsets (2), their sigma (2) and the seed map (1).
//!
//! Every convolution is followed by BatchNorm (eval mode: its running statistics) and ReLU,
//! except the output blocks (no activation). BatchNorm is folded into the convolution once, when
//! the weights load, so the network runs as plain convolutions. The block and channel counts are
//! read from the weights, not fixed here.

use std::collections::BTreeMap;

use burn::tensor::activation::relu;
use burn::tensor::module::{conv2d, interpolate, max_pool2d};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions, MaxPoolOptions};
use burn::tensor::{Device, Tensor, TensorData};

use crate::torchscript;

/// BatchNorm's epsilon in the published models.
const BN_EPS: f32 = 1e-5;

/// How much the input shrinks at the deepest level: sizes must be multiples of it (InstanSeg
/// pads to 16).
pub const DIVISOR: usize = 16;

/// A convolution with its BatchNorm folded in, and whether ReLU follows.
struct ConvBn {
    weight: Tensor<4>,
    bias: Tensor<1>,
    padding: usize,
    relu: bool,
}

impl ConvBn {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = conv2d(x, self.weight.clone(), Some(self.bias.clone()), ConvOptions::new([1, 1], [self.padding, self.padding], [1, 1], 1));
        if self.relu { relu(y) } else { y }
    }
}

/// `Some(conv)` for a `Sequential(Conv2d, BatchNorm2d | Identity, act)` at `prefix`, `None` when
/// the block has an identity there (InstanSeg's shallow blocks).
fn conv_bn(w: &BTreeMap<String, torchscript::Tensor>, prefix: &str, relu: bool, device: &Device) -> Result<Option<ConvBn>, String> {
    let Some(cw) = w.get(&format!("{prefix}.0.weight")) else { return Ok(None) };
    let [o, i, kh, kw] = <[usize; 4]>::try_from(cw.shape.clone()).map_err(|_| format!("{prefix}.0.weight: shape {:?}, expected 4 dimensions", cw.shape))?;
    if kh != kw || kh % 2 == 0 {
        return Err(format!("{prefix}: a {kh} × {kw} kernel"));
    }
    let mut weight = cw.data.clone();
    let mut bias = w.get(&format!("{prefix}.0.bias")).map_or(vec![0.0; o], |b| b.data.clone());
    if let Some(gamma) = w.get(&format!("{prefix}.1.weight")) {
        let get = |n: &str| w.get(&format!("{prefix}.1.{n}")).map(|t| t.data.clone()).ok_or_else(|| format!("{prefix}.1.{n} is missing"));
        let (beta, mean, var) = (get("bias")?, get("running_mean")?, get("running_var")?);
        let per = i * kh * kw;
        for c in 0..o {
            let s = gamma.data[c] / (var[c] + BN_EPS).sqrt();
            for v in &mut weight[c * per..(c + 1) * per] {
                *v *= s;
            }
            bias[c] = (bias[c] - mean[c]) * s + beta[c];
        }
    }
    Ok(Some(ConvBn {
        weight: Tensor::<4>::from_data(TensorData::new(weight, [o, i, kh, kw]), device),
        bias: Tensor::<1>::from_data(TensorData::new(bias, [o]), device),
        padding: kh / 2,
        relu,
    }))
}

fn need(c: Option<ConvBn>, name: &str) -> Result<ConvBn, String> {
    c.ok_or_else(|| format!("{name} is missing"))
}

/// `f` of `x`, or `x` itself for an identity.
fn apply(c: &Option<ConvBn>, x: Tensor<4>) -> Tensor<4> {
    match c {
        Some(c) => c.forward(x),
        None => x,
    }
}

struct EncoderBlock {
    pool: bool,
    conv0: ConvBn,
    conv1: ConvBn,
    conv2: Option<ConvBn>,
    conv3: Option<ConvBn>,
    conv4: ConvBn,
}

impl EncoderBlock {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = if self.pool { max_pool2d(x, MaxPoolOptions::new([2, 2]).with_stride([2, 2])) } else { x };
        let proj = self.conv0.forward(x.clone());
        let x = proj + apply(&self.conv2, self.conv1.forward(x));
        x.clone() + self.conv4.forward(apply(&self.conv3, x))
    }
}

struct DecoderBlock {
    conv0: ConvBn,
    conv_skip: ConvBn,
    conv1: ConvBn,
    conv2: ConvBn,
    conv3: Option<ConvBn>,
    conv4: ConvBn,
}

impl DecoderBlock {
    fn forward(&self, x: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = x.dims();
        let x = interpolate(x, InterpolateOptions::new(InterpolateMode::Nearest).with_output_size([h * 2, w * 2]));
        let proj = self.conv0.forward(x.clone());
        let x = self.conv1.forward(x);
        let x = proj + self.conv2.forward(x + self.conv_skip.forward(skip));
        x.clone() + self.conv4.forward(apply(&self.conv3, x))
    }
}

struct Decoder {
    blocks: Vec<DecoderBlock>,
    heads: Vec<ConvBn>,
}

pub struct InstanSegNet {
    encoder: Vec<EncoderBlock>,
    decoders: Vec<Decoder>,
    pub device: Device,
    /// Channels its images have.
    pub channels: usize,
    /// Channels it gives (embedding, sigma, seeds).
    pub outputs: usize,
}

impl InstanSegNet {
    /// The network on `device` from the archive's `fcn.*` weights.
    pub fn new(w: &BTreeMap<String, torchscript::Tensor>, device: Device) -> Result<Self, String> {
        let mut encoder = Vec::new();
        for i in 0.. {
            let p = format!("fcn.encoder.{i}");
            let Some(conv0) = conv_bn(w, &format!("{p}.conv0"), true, &device)? else { break };
            encoder.push(EncoderBlock {
                pool: i > 0,
                conv0,
                conv1: need(conv_bn(w, &format!("{p}.conv1"), true, &device)?, &format!("{p}.conv1"))?,
                conv2: conv_bn(w, &format!("{p}.conv2"), true, &device)?,
                conv3: conv_bn(w, &format!("{p}.conv3"), true, &device)?,
                conv4: need(conv_bn(w, &format!("{p}.conv4"), true, &device)?, &format!("{p}.conv4"))?,
            });
        }
        if encoder.len() < 2 {
            return Err(format!("{} encoder block(s) in the weights", encoder.len()));
        }
        let mut decoders = Vec::new();
        for d in 0.. {
            let p = format!("fcn.decoders.{d}");
            let mut blocks = Vec::new();
            for b in 0.. {
                let q = format!("{p}.decoder.{b}");
                let Some(conv0) = conv_bn(w, &format!("{q}.conv0"), true, &device)? else { break };
                blocks.push(DecoderBlock {
                    conv0,
                    conv_skip: need(conv_bn(w, &format!("{q}.conv_skip"), true, &device)?, &format!("{q}.conv_skip"))?,
                    conv1: need(conv_bn(w, &format!("{q}.conv1"), true, &device)?, &format!("{q}.conv1"))?,
                    conv2: need(conv_bn(w, &format!("{q}.conv2"), true, &device)?, &format!("{q}.conv2"))?,
                    conv3: conv_bn(w, &format!("{q}.conv3"), true, &device)?,
                    conv4: need(conv_bn(w, &format!("{q}.conv4"), true, &device)?, &format!("{q}.conv4"))?,
                });
            }
            if blocks.is_empty() {
                break;
            }
            if blocks.len() != encoder.len() - 1 {
                return Err(format!("{p}: {} decoder blocks for {} encoder blocks", blocks.len(), encoder.len()));
            }
            let mut heads = Vec::new();
            for k in 0.. {
                match conv_bn(w, &format!("{p}.final_block.{k}"), false, &device)? {
                    Some(c) => heads.push(c),
                    None => break,
                }
            }
            if heads.is_empty() {
                return Err(format!("{p}: no output blocks"));
            }
            decoders.push(Decoder { blocks, heads });
        }
        if decoders.is_empty() {
            return Err("no decoder in the weights".into());
        }
        let channels = w["fcn.encoder.0.conv0.0.weight"].shape[1];
        let outputs = decoders.iter().flat_map(|d| &d.heads).map(|h| h.bias.dims()[0]).sum();
        Ok(Self { encoder, decoders, device, channels, outputs })
    }

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
        let outs: Vec<Tensor<4>> = self
            .decoders
            .iter()
            .map(|d| {
                let mut y = x.clone();
                for (block, skip) in d.blocks.iter().zip(skips.iter().rev()) {
                    y = block.forward(y, skip.clone());
                }
                Tensor::cat(d.heads.iter().map(|h| h.forward(y.clone())).collect(), 1)
            })
            .collect();
        Tensor::cat(outs, 1)
    }
}
