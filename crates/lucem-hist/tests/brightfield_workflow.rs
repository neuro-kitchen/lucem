//! A Ki67-style analysis of a synthetic H-DAB tile, end to end: stains separated, cells detected
//! on hematoxylin, their DAB scored, the region summarized and exported.

use lucem_core::cubecl::prelude::Client;
use lucem_hist::cells::{detect_cells, CellDetection, CellImage};
use lucem_hist::export::{geojson, measurements_table, region_summary};
use lucem_hist::intensity::{classify, IntensityThresholds};
use lucem_hist::stains::{deconvolve, StainVectors};

const W: usize = 128;
const H: usize = 96;

/// Twelve nuclei on a 4 × 3 grid; the first four carry DAB (positive), weak to strong.
fn tile() -> (Vec<u8>, usize) {
    let s = StainVectors::h_dab();
    let mut rgb = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let (mut hem, mut dab) = (0.03, 0.01);
            for k in 0..12 {
                let (cx, cy) = (16.0 + (k % 4) as f64 * 32.0, 16.0 + (k / 4) as f64 * 32.0);
                if (x as f64 + 0.5 - cx).powi(2) + (y as f64 + 0.5 - cy).powi(2) <= 36.0 {
                    hem = 0.6;
                    if k < 4 {
                        dab = 0.2 + 0.2 * k as f64;
                    }
                }
            }
            for c in 0..3 {
                let od = hem * s.stains[0].od[c] + dab * s.stains[1].od[c];
                rgb[(y * W + x) * 3 + c] = (255.0 * 10f64.powf(-od)).round() as u8;
            }
        }
    }
    (rgb, 4)
}

fn ki67_style_analysis(client: &Client) {
    let (rgb, positives) = tile();
    let [hem, dab, _] = deconvolve(&rgb, &StainVectors::h_dab()).unwrap();
    let img = CellImage { width: W, height: H, detection: &hem, channels: vec![("Hematoxylin OD".into(), &hem), ("DAB OD".into(), &dab)], origin: [0.0, 0.0], downsample: 1.0, pixel_um: 0.5 };
    let s = CellDetection { background_radius_um: 0.0, sigma_um: 0.5, threshold: 0.3, min_area_um2: 5.0, max_area_um2: 100.0, cell_expansion_um: 2.0, ..CellDetection::default() };
    let mut cells = detect_cells(client, &img, &s).unwrap().objects;
    assert_eq!(cells.len(), 12, "{}", client.name());
    classify(&mut cells, &IntensityThresholds { measurement: "Nucleus: DAB OD mean".into(), thresholds: vec![0.1, 0.35, 0.55], filter: None });
    let region = vec![[0.0, 0.0], [W as f64, 0.0], [W as f64, H as f64], [0.0, H as f64]];
    let summary = region_summary(&region, &cells, 0.5);
    let want = 100.0 * positives as f64 / 12.0;
    assert!((summary["Positive %"] - want).abs() < 1e-9, "{}: {summary:?}", client.name());
    assert_eq!((summary["Num 1+"], summary["Num 2+"], summary["Num 3+"]), (1.0, 1.0, 2.0), "{}: {summary:?}", client.name());
    let table = measurements_table(&cells, "tile", 0.5);
    assert_eq!(table.rows.len(), 12);
    assert!(table.header.iter().any(|h| h == "Nucleus: DAB OD mean"));
    assert_eq!(geojson(&cells)["features"].as_array().unwrap().len(), 12);
}

lucem_core::runtime_test!(a_ki67_style_analysis_runs_end_to_end, ki67_style_analysis);
