//! Publication figure data model (MVVM Model layer): artboard layout, panels, sources, and
//! calibrated scale-bar math.

use std::sync::Arc;

use lucem_io::Pyramid;

use crate::domain::graph::NodeKind;
use crate::domain::project::ChannelDoc;

/// Artboard publication background.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FigureBg {
    Dark,
    White,
    Black,
}

impl FigureBg {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::White => "White",
            Self::Black => "Black",
        }
    }

    pub fn rgb(self) -> [u8; 3] {
        match self {
            Self::Dark => [18, 20, 26],
            Self::White => [255, 255, 255],
            Self::Black => [0, 0, 0],
        }
    }

    pub fn hex(self) -> &'static str {
        match self {
            Self::Dark => "#12141a",
            Self::White => "#ffffff",
            Self::Black => "#000000",
        }
    }

    pub fn is_light(self) -> bool {
        matches!(self, Self::White)
    }

    pub fn text_hex(self) -> &'static str {
        if self.is_light() { "#111318" } else { "#f3f5f8" }
    }

    pub fn muted_hex(self) -> &'static str {
        if self.is_light() { "#5a6270" } else { "#9aa3b2" }
    }
}

/// Which corner of a panel holds its calibrated scale bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleCorner {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
}

impl ScaleCorner {
    pub const ALL: [Self; 4] = [Self::BottomLeft, Self::BottomRight, Self::TopLeft, Self::TopRight];

    pub fn short(self) -> &'static str {
        match self {
            Self::BottomLeft => "BL",
            Self::BottomRight => "BR",
            Self::TopLeft => "TL",
            Self::TopRight => "TR",
        }
    }
}

/// What a figure panel renders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PanelSource {
    /// Multi-channel merge of all visible channels of `image`.
    Composite { image: String },
    /// Single channel of `image`.
    Channel { image: String, channel: usize },
    /// Computed Pipeline node stage/preview of `image`.
    PipelineStage { image: String, node_id: String, node_label: String },
}

impl PanelSource {
    pub fn image_id(&self) -> &str {
        match self {
            Self::Composite { image } | Self::Channel { image, .. } | Self::PipelineStage { image, .. } => image,
        }
    }
}

/// One panel on the publication artboard.
#[derive(Debug, Clone)]
pub struct FigurePanel {
    pub id: usize,
    pub title: String,
    pub subtitle: String,
    pub source: PanelSource,
    /// Optional level-0 crop `[x, y, w, h]`. `None` = full slide.
    pub region: Option<[f64; 4]>,
    pub region_label: String,
    pub show_scale_bar: bool,
    /// Manual scale bar length in µm (`None` = auto nice round value).
    pub scale_bar_um: Option<f64>,
    pub scale_corner: ScaleCorner,
}

/// Pure document state for a multi-panel publication figure.
#[derive(Debug, Clone)]
pub struct FigureDoc {
    pub title: String,
    pub subtitle: String,
    pub bg: FigureBg,
    pub cols: usize,
    pub gap_px: usize,
    pub show_letters: bool,
    pub panels: Vec<FigurePanel>,
    pub selected: Option<usize>,
    pub next_id: usize,
}

impl Default for FigureDoc {
    fn default() -> Self {
        Self {
            title: "Figure 1".into(),
            subtitle: String::new(),
            bg: FigureBg::Dark,
            cols: 3,
            gap_px: 12,
            show_letters: true,
            panels: Vec::new(),
            selected: None,
            next_id: 1,
        }
    }
}

impl FigureDoc {
    pub fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Zooms `panel.region` in (`factor < 1.0`) or out (`factor > 1.0`) around its center, clamped to
/// the slide bounds `(iw, ih)`. Zooming all the way out resets to `Full slide`.
pub fn zoom_panel_region(panel: &mut FigurePanel, factor: f64, iw: f64, ih: f64) {
    let [x, y, w, h] = panel.region.unwrap_or([0.0, 0.0, iw, ih]);
    let cx = x + w * 0.5;
    let cy = y + h * 0.5;
    let nw = (w * factor).clamp(64.0, iw);
    let nh = (h * factor).clamp(64.0, ih);
    if nw >= iw * 0.99 && nh >= ih * 0.99 {
        panel.region = None;
        panel.region_label = "Full slide".into();
        return;
    }
    let nx = (cx - nw * 0.5).clamp(0.0, (iw - nw).max(0.0));
    let ny = (cy - nh * 0.5).clamp(0.0, (ih - nh).max(0.0));
    panel.region = Some([nx, ny, nw, nh]);
    let mag = (iw / nw).max(1.0);
    panel.region_label = format!("Custom crop ({mag:.1}×)");
}

/// Pans `panel.region` by `(dx_frac, dy_frac)` of its current crop width/height.
pub fn pan_panel_region(panel: &mut FigurePanel, dx_frac: f64, dy_frac: f64, iw: f64, ih: f64) {
    let Some([x, y, w, h]) = panel.region else { return };
    let nx = (x + dx_frac * w).clamp(0.0, (iw - w).max(0.0));
    let ny = (y + dy_frac * h).clamp(0.0, (ih - h).max(0.0));
    panel.region = Some([nx, ny, w, h]);
}

pub fn panel_letter(idx: usize) -> String {
    if idx < 26 {
        ((b'A' + idx as u8) as char).to_string()
    } else {
        format!("P{}", idx + 1)
    }
}

pub fn panel_render_key(p: &FigurePanel, channels: &[ChannelDoc], z: usize) -> String {
    let reg = match p.region {
        Some([x, y, w, h]) => format!("{x:.0},{y:.0},{w:.0},{h:.0}"),
        None => "full".into(),
    };
    let ch_sig: String = channels
        .iter()
        .map(|c| format!("{}:{:.2}:{:.2}:{:02x}{:02x}{:02x}", c.visible as u8, c.min, c.max, c.color[0], c.color[1], c.color[2]))
        .collect::<Vec<_>>()
        .join("|");
    format!("{:?}|{reg}|z{z}|{ch_sig}", p.source)
}

/// Computes `(fraction_of_panel_width, formatted_label)` for a panel's calibrated scale bar.
pub fn panel_scale_bar(p: &FigurePanel, pyr: Option<&Arc<Pyramid>>) -> Option<(f64, String)> {
    let pyr = pyr?;
    let um_per_px = pyr.pixel_size_um?;
    if um_per_px <= 0.0 {
        return None;
    }
    let width_px = p.region.map_or(pyr.width() as f64, |r| r[2]);
    let width_um = (width_px * um_per_px).max(1e-3);
    let len_um = p.scale_bar_um.unwrap_or_else(|| nice_scale_um(width_um * 0.22));
    let frac = (len_um / width_um).clamp(0.05, 0.55);
    let label = if len_um >= 1000.0 { format!("{:.0} mm", len_um / 1000.0) } else { format!("{:.0} µm", len_um) };
    Some((frac, label))
}

pub fn nice_scale_um(target: f64) -> f64 {
    const STEPS: [f64; 12] = [5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0];
    STEPS.iter().copied().min_by(|a, b| (a - target).abs().total_cmp(&(b - target).abs())).unwrap_or(100.0)
}

pub fn node_stage_label(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Source { channel, .. } => format!("Source ch {}", channel + 1),
        NodeKind::Annotations { class, .. } => class.clone().unwrap_or_else(|| "Annotations".into()),
        NodeKind::Op { op, .. } => format!("{op:?}").split('{').next().unwrap_or("Op").trim().to_string(),
        NodeKind::View { name, .. } => name.clone(),
        NodeKind::Script { path, .. } => std::path::Path::new(path).file_name().and_then(|s| s.to_str()).unwrap_or("Script").to_string(),
        NodeKind::Model { model, .. } => model.clone(),
    }
}

pub fn sanitize_filename(title: &str, ext: &str) -> String {
    let stem: String = title.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    let stem = stem.trim_matches('_');
    let stem = if stem.is_empty() { "figure" } else { stem };
    format!("{stem}.{ext}")
}
