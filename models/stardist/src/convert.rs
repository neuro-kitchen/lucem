//! The published Keras weights (`stardist_weights.h5`, HDF5) as safetensors: every convolution's
//! kernel turned from Keras' `[kh, kw, in, out]` to Burn's `[out, in, kh, kw]`, named
//! `<layer>.weight` and `<layer>.bias` — the paths of [`crate::net::StarDistModule`]'s fields, so
//! Burn's `SafetensorsStore` loads the file as it is; the source and its hash in the metadata. Read with a
//! pure-Rust HDF5 decoder, so neither Python nor the HDF5 C library is needed.

use std::collections::HashMap;
use std::path::Path;

use hdf5_reader::Hdf5File;
use safetensors::tensor::{Dtype, TensorView};

use crate::net::layers;

/// One layer's weights in Burn's layout.
pub struct LayerWeights {
    pub name: &'static str,
    /// `[out, in, kh, kw]`.
    pub shape: [usize; 4],
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Reads every layer of the network (for `channels`-channel images) from the Keras file,
/// checking each shape.
pub fn read_keras(path: &Path, channels: usize) -> Result<Vec<LayerWeights>, String> {
    let f = Hdf5File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    layers(channels)
        .iter()
        .map(|l| {
            let kernel = f.dataset(&format!("/{0}/{0}/kernel:0", l.name)).map_err(|e| format!("{}: kernel: {e}", l.name))?;
            let bias = f.dataset(&format!("/{0}/{0}/bias:0", l.name)).map_err(|e| format!("{}: bias: {e}", l.name))?;
            let s: Vec<usize> = kernel.shape().iter().map(|&d| d as usize).collect();
            let want = [l.kernel, l.kernel, l.input, l.output];
            if s != want {
                return Err(format!("{}: kernel {s:?}, expected {want:?}", l.name));
            }
            let k: Vec<f32> = kernel.read_array::<f32>().map_err(|e| format!("{}: {e}", l.name))?.iter().copied().collect();
            let b: Vec<f32> = bias.read_array::<f32>().map_err(|e| format!("{}: {e}", l.name))?.iter().copied().collect();
            if b.len() != l.output {
                return Err(format!("{}: {} biases, expected {}", l.name, b.len(), l.output));
            }
            // [kh, kw, in, out] → [out, in, kh, kw]
            let (kh, kw, ci, co) = (l.kernel, l.kernel, l.input, l.output);
            let mut w = vec![0f32; k.len()];
            for y in 0..kh {
                for x in 0..kw {
                    for i in 0..ci {
                        for o in 0..co {
                            w[((o * ci + i) * kh + y) * kw + x] = k[((y * kw + x) * ci + i) * co + o];
                        }
                    }
                }
            }
            Ok(LayerWeights { name: l.name, shape: [co, ci, kh, kw], weight: w, bias: b })
        })
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Writes the layers as safetensors with `metadata` (source, hash, licence).
pub fn write_safetensors(layers: &[LayerWeights], metadata: HashMap<String, String>, out: &Path) -> Result<(), String> {
    let data: Vec<(String, Vec<usize>, Vec<u8>)> = layers
        .iter()
        .flat_map(|l| [(format!("{}.weight", l.name), l.shape.to_vec(), bytes(&l.weight)), (format!("{}.bias", l.name), vec![l.bias.len()], bytes(&l.bias))])
        .collect();
    let views: Vec<(String, TensorView)> = data.iter().map(|(n, s, b)| TensorView::new(Dtype::F32, s.clone(), b).map(|v| (n.clone(), v)).map_err(|e| format!("{n}: {e}"))).collect::<Result<_, _>>()?;
    let tmp = out.with_extension("tmp");
    safetensors::tensor::serialize_to_file(views, Some(metadata), &tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))
}
