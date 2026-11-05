//! Executes a chain of [`ModifierNode`]s on the active CubeCL [`Client`] and computes live
//! Arrow/Polars analytics, Marching Squares vector contours, and summary statistics.

use std::time::Instant;

use arrow::record_batch::RecordBatch;
use lumen_core::cubecl::prelude::Client;
use lumen_core::memory::DeviceArray;
use lumen_ops::analytics::{histogram_to_dataframe, regions_to_dataframe, regions_to_record_batch};
use lumen_ops::core::{elementwise, histogram, reduce, StructuringElement};
use lumen_ops::error::OpsResult;
use lumen_ops::feature::canny::{canny, Canny};
use lumen_ops::filter::gaussian::{gaussian_blur, GaussianBlur};
use lumen_ops::filter::gradient::{sobel, GradientFilter};
use lumen_ops::intensity::contrast::adjust_gamma;
use lumen_ops::intensity::equalize::{clahe, ClaheSettings};
use lumen_ops::measure::contour::{marching_squares, Contour2d};
use lumen_ops::measure::regions::{label_components, region_features, RegionFeatures};
use lumen_ops::measure::topology::{blur_score, shannon_entropy};
use lumen_ops::morphology::grayscale::{morph_gradient, GrayMorphology};
use lumen_ops::restore::background::{rolling_ball, RollingBall};
use lumen_ops::restore::denoise::{tv_chambolle, TvChambolle};
use lumen_ops::threshold::global::otsu;
use polars::prelude::DataFrame;

use crate::modifier::{ModifierKind, ModifierNode};

/// Complete output of evaluating the non-destructive GPU pipeline + analytics pass.
#[derive(Debug, Clone)]
pub struct PipelineExecutionResult {
    pub width: usize,
    pub height: usize,
    /// Processed `[height, width]` image on the host for display tonemapping & pixel inspection.
    pub output_host: Vec<f32>,
    /// Global min, max, mean, and std of the processed image.
    pub min_val: f64,
    pub max_val: f64,
    pub mean_val: f64,
    pub std_val: f64,
    pub otsu_level: f64,
    pub entropy_bits: f64,
    pub sharpness_score: f64,
    /// 64-bin histogram counts and centers for the Analytics chart.
    pub hist_centers: Vec<f64>,
    pub hist_counts: Vec<u32>,
    pub hist_df: DataFrame,
    /// Marching Squares iso-contours at the Otsu threshold level (for VectorCraft contour overlay).
    pub contours: Vec<Contour2d>,
    /// Extracted connected-component region features + Polars DataFrame + Arrow RecordBatch.
    pub regions: Vec<RegionFeatures>,
    pub regions_df: DataFrame,
    pub regions_batch: RecordBatch,
    /// Total GPU pipeline execution time in milliseconds.
    pub elapsed_ms: f64,
}

pub struct PipelineExecutor;

impl PipelineExecutor {
    /// Runs all enabled `modifiers` sequentially on `input_dev` on the GPU, then extracts
    /// histogram, Marching Squares contours, and connected-component `RegionFeatures` into
    /// Polars `DataFrame`s and Arrow `RecordBatch`es.
    pub fn execute(
        client: &Client,
        input_dev: &DeviceArray<f32>,
        modifiers: &[ModifierNode],
        extract_contours: bool,
    ) -> OpsResult<PipelineExecutionResult> {
        let t0 = Instant::now();
        let dims = input_dev.dims();
        let (height, width) = (dims[0], dims[1]);

        let mut current = elementwise::copy(client, input_dev)?;

        for node in modifiers.iter().filter(|m| m.enabled) {
            current = match node.kind {
                ModifierKind::GaussianBlur => {
                    let s = node.primary_param.clamp(0.1, 25.0);
                    gaussian_blur(client, &current, &[s, s], &GaussianBlur::default())?
                }
                ModifierKind::SobelEdges => {
                    let mag = sobel(client, &current, &GradientFilter::default())?;
                    if (node.primary_param - 1.0).abs() > 1e-3 {
                        let scaled = DeviceArray::<f32>::empty(client, mag.shape().clone())?;
                        elementwise::scale_offset_into(client, &mag, &scaled, node.primary_param, 0.0)?;
                        scaled
                    } else {
                        mag
                    }
                }
                ModifierKind::ClaheContrast => {
                    let clip = node.primary_param.clamp(0.002, 0.5);
                    let grid = (node.secondary_param.round() as usize).clamp(2, 16);
                    clahe(
                        client,
                        &current,
                        ClaheSettings {
                            grid_size: (grid, grid),
                            clip_limit: clip,
                            num_bins: 128,
                        },
                    )?
                }
                ModifierKind::GammaCurve => {
                    let gamma = node.primary_param.clamp(0.1, 5.0);
                    let gain = node.secondary_param.clamp(0.1, 5.0);
                    adjust_gamma(client, &current, gamma, gain)?
                }
                ModifierKind::TvChambolleDenoise => {
                    let weight = node.primary_param.clamp(0.01, 2.0);
                    let iters = (node.secondary_param.round() as usize).clamp(5, 100);
                    tv_chambolle(
                        client,
                        &current,
                        &TvChambolle {
                            weight,
                            eps: 2e-4,
                            max_num_iter: iters,
                            isotropic: true,
                        },
                    )?
                }
                ModifierKind::RollingBallBackground => {
                    let radius = node.primary_param.clamp(1.0, 64.0);
                    let bg = rolling_ball(
                        client,
                        &current,
                        &RollingBall {
                            radius,
                            ..RollingBall::default()
                        },
                    )?;
                    // Subtract rolling-ball background from current
                    let sub = DeviceArray::<f32>::empty(client, current.shape().clone())?;
                    elementwise::linear_combination_into(client, &current, &bg, &sub, 1.0, -1.0, 0.0)?;
                    let clipped = DeviceArray::<f32>::empty(client, current.shape().clone())?;
                    elementwise::clip_into(client, &sub, &clipped, 0.0, 1.0)?;
                    clipped
                }
                ModifierKind::CannyEdges => {
                    let sigma = node.primary_param.clamp(0.3, 10.0);
                    let edges_u32 = canny(
                        client,
                        &current,
                        &Canny {
                            sigma,
                            ..Canny::default()
                        },
                    )?;
                    elementwise::cast::<u32, f32>(client, &edges_u32)?
                }
                ModifierKind::OtsuThreshold => {
                    let bins = (node.primary_param.round() as usize).clamp(16, 512);
                    let t = otsu(client, &current, bins)?;
                    let mask = DeviceArray::<u32>::empty(client, current.shape().clone())?;
                    elementwise::threshold_scalar_into(client, &current, t, true, &mask)?;
                    elementwise::cast::<u32, f32>(client, &mask)?
                }
                ModifierKind::MorphGradient => {
                    let r = (node.primary_param.round() as usize).clamp(1, 10);
                    let se = StructuringElement::disk(r);
                    morph_gradient(client, &current, &se, &GrayMorphology::default())?
                }
            };
        }

        let s = reduce::sum_min_max(client, &current)?;
        let (mean_val, var_val) = reduce::mean_var(client, &current, 0)?;
        let std_val = var_val.max(0.0).sqrt();

        let (counts_dev, hist_centers) = histogram::histogram(client, &current, 64, None)?;
        let hist_counts = counts_dev.to_host(client);
        let (_, hist_df) = histogram_to_dataframe(&hist_centers, &hist_counts)?;

        let otsu_level = otsu(client, &current, 64).unwrap_or(0.5 * (s.min + s.max));
        let entropy_bits = shannon_entropy(client, &current, 64, 2.0).unwrap_or(0.0);
        let sharpness_score = blur_score(client, &current).unwrap_or(0.0);

        let contours = if extract_contours {
            marching_squares(client, &current, otsu_level).unwrap_or_default()
        } else {
            Vec::new()
        };

        let bin_mask = DeviceArray::<u32>::empty(client, current.shape().clone())?;
        elementwise::threshold_scalar_into(client, &current, otsu_level, true, &bin_mask)?;
        let cc = label_components(client, &bin_mask, None)?;
        let mut regions = if cc.num_labels <= 8192 {
            region_features(client, &cc.labels, Some(&current)).unwrap_or_default()
        } else {
            Vec::new()
        };
        // Filter out single-pixel noise speckles so the RegionProps table shows meaningful objects
        regions.retain(|r| r.area >= 4);
        let regions_df = regions_to_dataframe(&regions)?;
        let regions_batch = regions_to_record_batch(&regions)?;

        let output_host = current.to_host(client);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(PipelineExecutionResult {
            width,
            height,
            output_host,
            min_val: s.min,
            max_val: s.max,
            mean_val,
            std_val,
            otsu_level,
            entropy_bits,
            sharpness_score,
            hist_centers,
            hist_counts,
            hist_df,
            contours,
            regions,
            regions_df,
            regions_batch,
            elapsed_ms,
        })
    }
}
