//! `lucem-hist`: the histology workflows of lucem, as a library the app, the command line and
//! scripts share. Each module is one step of a whole-slide analysis:
//!
//! | module | what it does |
//! |---|---|
//! | [`stains`] | stain vectors (presets, estimated from a region, background) and colour deconvolution |
//! | [`tissue`] | tissue detection on a low-resolution image: smoothed threshold, small pieces and holes removed |
//! | [`area`] | areas of classes and of positive staining inside a region |
//! | [`cells`] | nucleus detection (background, smoothing, threshold, split by shape), cell expansion, measurements per compartment and channel (GPU, through `lucem-ops`); cells from a model's nucleus labels too |
//! | [`features`] | smoothed features (a Gaussian-weighted mean of the neighbours' measurements) |
//! | [`classify`] | object classifiers: one measurement and a threshold, Random Trees trained from class-labelled regions, composites; saved as JSON |
//! | [`intensity`] | intensity classes (negative, 1+, 2+, 3+), Positive % and H-score |
//! | [`density`] | density maps of objects, their hotspots and thresholded regions |
//! | [`multiplex`] | per-marker classifiers combined, phenotypes from marker combinations |
//! | [`pixel`] | pixel classifiers: multiscale features (GPU) and Random Trees, class maps, regions and areas |
//! | [`slide`] | whole regions tile by tile (overlap; each cell once, by the tile whose core holds it), inside a drawn region; brightfield or fluorescence input |
//! | [`superpixel`] | SLIC superpixels as tile objects with their measurements |
//! | [`export`] | measurement tables (TSV / CSV), region summaries, GeoJSON |
//! | [`models`] | histology models as plugins (`plugins/ml/lucem-ml-<family>`): the families, their tasks and licences |
//!
//! Coordinates of objects are level-0 pixels of the slide; lengths and areas in measurements are
//! micrometres (from the image's pixel size), as the measurement names say.

pub mod area;
pub mod cells;
pub mod classify;
pub mod density;
pub mod engine;
pub mod error;
pub mod export;
pub mod features;
pub mod forest;
pub mod intensity;
pub mod models;
pub mod multiplex;
pub mod objects;
pub mod output;
pub mod pixel;
pub mod polygon;
pub mod raster;
pub mod session;
pub mod slide;
pub mod stains;
pub mod superpixel;
pub mod tissue;

pub use error::{HistError, HistResult};
pub use objects::{HistObject, IntensityClass, Measurements, ObjectKind, Point, Polygon};
