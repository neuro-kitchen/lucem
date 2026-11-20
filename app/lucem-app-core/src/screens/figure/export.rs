//! Panel rasterization and publication PNG / SVG export engine for the Figure composer.

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::RenderImage;
use lucem_io::Pyramid;

use crate::domain::figure::{panel_letter, panel_scale_bar, FigureDoc, PanelSource, ScaleCorner};
use crate::domain::project::ChannelDoc;

/// Cached rendered panel image (both GPUI texture and raw RGB for PNG/SVG export).
#[derive(Clone)]
pub struct RenderedPanel {
    pub key: String,
    pub image: Arc<RenderImage>,
    pub rgb: Arc<Vec<u8>>,
    pub w: usize,
    pub h: usize,
}

/// Renders one panel's crop into an RGB8 buffer at ~512 px max side using the exact channel LUTs.
pub fn render_panel_rgb(
    pyr: &Pyramid,
    channels_doc: &[ChannelDoc],
    source: &PanelSource,
    region: Option<[f64; 4]>,
    z: usize,
) -> Option<(usize, usize, Vec<u8>)> {
    let (iw, ih) = (pyr.width() as f64, pyr.height() as f64);
    let [rx, ry, rw, rh] = region.unwrap_or([0.0, 0.0, iw, ih]);
    let target_side = 512.0;
    let want_ds = (rw.max(rh) / target_side).max(1.0);
    let level = pyr
        .levels
        .iter()
        .enumerate()
        .rev()
        .find(|(_, l)| l.downsample <= want_ds)
        .map(|(i, _)| i)
        .unwrap_or(pyr.levels.len().saturating_sub(1));
    let l = &pyr.levels[level];
    let ds = l.downsample;
    let x0 = ((rx / ds).floor().max(0.0) as usize).min(l.width.saturating_sub(1));
    let y0 = ((ry / ds).floor().max(0.0) as usize).min(l.height.saturating_sub(1));
    let x1 = (((rx + rw) / ds).ceil().max(1.0) as usize).min(l.width);
    let y1 = (((ry + rh) / ds).ceil().max(1.0) as usize).min(l.height);
    let (w, h) = (x1.saturating_sub(x0).max(1), y1.saturating_sub(y0).max(1));
    let plane = ((z as f64 / l.z_downsample) as usize).min(l.depth.saturating_sub(1));

    let chosen: Vec<usize> = match source {
        PanelSource::Channel { channel, .. } => vec![*channel],
        PanelSource::Composite { .. } | PanelSource::PipelineStage { .. } => {
            let vis: Vec<usize> = channels_doc.iter().enumerate().filter(|(_, c)| c.visible).map(|(i, _)| i).collect();
            if vis.is_empty() { (0..pyr.channels.len().min(4)).collect() } else { vis }
        }
    };

    let mut acc = vec![0.0f32; w * h * 3];
    for c in chosen {
        let Some(doc) = channels_doc.get(c) else { continue };
        let Ok(raw) = pyr.read_region_at(level, c, plane, x0, y0, w, h) else { continue };
        let span = (doc.max - doc.min).max(1e-12);
        let gamma = doc.gamma.clamp(0.1, 5.0);
        let [cr, cg, cb] = [doc.color[0] as f32 / 255.0, doc.color[1] as f32 / 255.0, doc.color[2] as f32 / 255.0];
        for (i, &v) in raw.iter().enumerate() {
            let t = ((v - doc.min) / span).clamp(0.0, 1.0).powf(gamma);
            let dst = i * 3;
            acc[dst] = (acc[dst] + t * cr).min(1.0);
            acc[dst + 1] = (acc[dst + 1] + t * cg).min(1.0);
            acc[dst + 2] = (acc[dst + 2] + t * cb).min(1.0);
        }
    }
    let rgb: Vec<u8> = acc.into_iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    Some((w, h, rgb))
}

pub fn render_image_to_rgb(img: &RenderImage) -> (usize, usize, Vec<u8>) {
    let sz = img.size(0);
    let (w, h) = (sz.width.0.max(1) as usize, sz.height.0.max(1) as usize);
    let mut rgb = vec![0u8; w * h * 3];
    if let Some(bytes) = img.as_bytes(0) {
        for (src, dst) in bytes.chunks_exact(4).zip(rgb.chunks_exact_mut(3)) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
        }
    }
    (w, h, rgb)
}

/// Composites the artboard into a single high-resolution RGB8 buffer (for PNG export).
pub fn compose_artboard_rgb(
    doc: &FigureDoc,
    renders: &HashMap<usize, RenderedPanel>,
    pyramids: &HashMap<String, Arc<Pyramid>>,
) -> Option<(usize, usize, Vec<u8>)> {
    if doc.panels.is_empty() {
        return None;
    }
    let (cell_w, cell_h, pad) = (512usize, 512usize, 32usize);
    let gap = doc.gap_px.max(4) * 2;
    let cols = doc.cols.clamp(1, 4);
    let rows = doc.panels.len().div_ceil(cols);
    let out_w = pad * 2 + cols * cell_w + cols.saturating_sub(1) * gap;
    let out_h = pad * 2 + rows * cell_h + rows.saturating_sub(1) * gap;
    let bg = doc.bg.rgb();
    let mut out = vec![0u8; out_w * out_h * 3];
    for px in out.chunks_exact_mut(3) {
        px.copy_from_slice(&bg);
    }
    for (idx, p) in doc.panels.iter().enumerate() {
        let Some(r) = renders.get(&p.id) else { continue };
        let ox = pad + (idx % cols) * (cell_w + gap);
        let oy = pad + (idx / cols) * (cell_h + gap);
        let scale = (cell_w as f64 / r.w.max(1) as f64).min(cell_h as f64 / r.h.max(1) as f64);
        let dw = ((r.w as f64 * scale).round() as usize).clamp(1, cell_w);
        let dh = ((r.h as f64 * scale).round() as usize).clamp(1, cell_h);
        let dx0 = ox + (cell_w - dw) / 2;
        let dy0 = oy + (cell_h - dh) / 2;
        for dy in 0..dh {
            let sy = (dy * r.h / dh).min(r.h - 1);
            for dx in 0..dw {
                let sx = (dx * r.w / dw).min(r.w - 1);
                let src = (sy * r.w + sx) * 3;
                let dst = ((dy0 + dy) * out_w + (dx0 + dx)) * 3;
                out[dst..dst + 3].copy_from_slice(&r.rgb[src..src + 3]);
            }
        }
        if p.show_scale_bar {
            if let Some((frac, _)) = panel_scale_bar(p, pyramids.get(p.source.image_id())) {
                let bar_w = ((dw as f64 * frac).round() as usize).clamp(12, dw / 2);
                let (bar_h, margin) = (6usize, 16usize);
                let bx = match p.scale_corner {
                    ScaleCorner::BottomLeft | ScaleCorner::TopLeft => dx0 + margin,
                    ScaleCorner::BottomRight | ScaleCorner::TopRight => (dx0 + dw).saturating_sub(margin + bar_w),
                };
                let by = match p.scale_corner {
                    ScaleCorner::TopLeft | ScaleCorner::TopRight => dy0 + margin,
                    ScaleCorner::BottomLeft | ScaleCorner::BottomRight => (dy0 + dh).saturating_sub(margin + bar_h),
                };
                for y in by..(by + bar_h).min(out_h) {
                    for x in bx..(bx + bar_w).min(out_w) {
                        let dst = (y * out_w + x) * 3;
                        out[dst..dst + 3].copy_from_slice(&[255, 255, 255]);
                    }
                }
            }
        }
    }
    Some((out_w, out_h, out))
}

/// Generates a publication-grade vector SVG with embedded PNG panels, vector titles, and scale bars.
pub fn compose_artboard_svg(
    doc: &FigureDoc,
    renders: &HashMap<usize, RenderedPanel>,
    pyramids: &HashMap<String, Arc<Pyramid>>,
) -> Option<String> {
    if doc.panels.is_empty() {
        return None;
    }
    let (cell_w, cell_h, title_h, pad) = (420usize, 420usize, 34usize, 28usize);
    let header_h = if doc.title.is_empty() && doc.subtitle.is_empty() { 0 } else { 58 };
    let gap = doc.gap_px.max(6) * 2;
    let cols = doc.cols.clamp(1, 4);
    let rows = doc.panels.len().div_ceil(cols);
    let out_w = pad * 2 + cols * cell_w + cols.saturating_sub(1) * gap;
    let out_h = pad * 2 + header_h + rows * (cell_h + title_h) + rows.saturating_sub(1) * gap;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {out_w} {out_h}">
  <rect width="100%" height="100%" fill="{}"/>
  <style>
    .fig-title {{ font: 600 20px system-ui, -apple-system, sans-serif; fill: {}; }}
    .fig-sub {{ font: 400 13px system-ui, -apple-system, sans-serif; fill: {}; }}
    .panel-title {{ font: 600 14px system-ui, -apple-system, sans-serif; fill: {}; }}
    .panel-sub {{ font: 400 11px system-ui, -apple-system, sans-serif; fill: {}; }}
    .scale-label {{ font: 600 11px system-ui, -apple-system, sans-serif; fill: #ffffff; }}
  </style>
"#,
        doc.bg.hex(),
        doc.bg.text_hex(),
        doc.bg.muted_hex(),
        doc.bg.text_hex(),
        doc.bg.muted_hex()
    );
    if header_h > 0 {
        svg.push_str(&format!("  <text x=\"{pad}\" y=\"{}\" class=\"fig-title\">{}</text>\n", pad + 20, xml_escape(&doc.title)));
        if !doc.subtitle.is_empty() {
            svg.push_str(&format!("  <text x=\"{pad}\" y=\"{}\" class=\"fig-sub\">{}</text>\n", pad + 42, xml_escape(&doc.subtitle)));
        }
    }
    for (idx, p) in doc.panels.iter().enumerate() {
        let ox = pad + (idx % cols) * (cell_w + gap);
        let oy = pad + header_h + (idx / cols) * (cell_h + title_h + gap);
        let img_y = oy + title_h;
        let heading = if doc.show_letters { format!("{}  {}", panel_letter(idx), p.title) } else { p.title.clone() };
        svg.push_str(&format!("  <text x=\"{ox}\" y=\"{}\" class=\"panel-title\">{}</text>\n", oy + 14, xml_escape(&heading)));
        if !p.subtitle.is_empty() {
            svg.push_str(&format!("  <text x=\"{ox}\" y=\"{}\" class=\"panel-sub\">{}</text>\n", oy + 28, xml_escape(&p.subtitle)));
        }
        if let Some(r) = renders.get(&p.id) {
            let mut png = Vec::new();
            if image::ImageEncoder::write_image(image::codecs::png::PngEncoder::new(&mut png), &r.rgb, r.w as u32, r.h as u32, image::ExtendedColorType::Rgb8).is_ok() {
                let b64 = base64_encode(&png);
                svg.push_str(&format!("  <image x=\"{ox}\" y=\"{img_y}\" width=\"{cell_w}\" height=\"{cell_h}\" preserveAspectRatio=\"xMidYMid meet\" href=\"data:image/png;base64,{b64}\"/>\n"));
            }
        }
        if p.show_scale_bar {
            if let Some((frac, label)) = panel_scale_bar(p, pyramids.get(p.source.image_id())) {
                let bar_w = ((cell_w as f64 * frac).round() as usize).clamp(16, cell_w / 2);
                let margin = 14usize;
                let bx = match p.scale_corner {
                    ScaleCorner::BottomLeft | ScaleCorner::TopLeft => ox + margin,
                    ScaleCorner::BottomRight | ScaleCorner::TopRight => (ox + cell_w).saturating_sub(margin + bar_w),
                };
                let by = match p.scale_corner {
                    ScaleCorner::TopLeft | ScaleCorner::TopRight => img_y + margin + 14,
                    ScaleCorner::BottomLeft | ScaleCorner::BottomRight => (img_y + cell_h).saturating_sub(margin + 4),
                };
                svg.push_str(&format!(
                    "  <rect x=\"{bx}\" y=\"{by}\" width=\"{bar_w}\" height=\"4\" rx=\"1\" fill=\"#ffffff\"/>\n  <text x=\"{bx}\" y=\"{}\" class=\"scale-label\">{}</text>\n",
                    by.saturating_sub(5),
                    xml_escape(&label)
                ));
            }
        }
    }
    svg.push_str("</svg>\n");
    Some(svg)
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = ((chunk[0] as u32) << 16)
            | ((chunk.get(1).copied().unwrap_or(0) as u32) << 8)
            | (chunk.get(2).copied().unwrap_or(0) as u32);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { TABLE[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[(n & 63) as usize] as char } else { '=' });
    }
    out
}

#[cfg(target_family = "wasm")]
pub fn save_or_download(filename: &str, mime: &str, bytes: &[u8], _folder: Option<std::path::PathBuf>) -> Result<String, String> {
    use wasm_bindgen::JsCast as _;
    let arr = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::of1(&arr);
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &bag).map_err(|_| "Blob creation failed".to_string())?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|_| "Object URL failed".to_string())?;
    let doc = web_sys::window().and_then(|w| w.document()).ok_or_else(|| "No browser document".to_string())?;
    let el = doc.create_element("a").map_err(|_| "Anchor creation failed".to_string())?;
    let a: web_sys::HtmlAnchorElement = el.dyn_into().map_err(|_| "Not an anchor".to_string())?;
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(format!("Downloaded {filename}"))
}

#[cfg(not(target_family = "wasm"))]
pub fn save_or_download(filename: &str, _mime: &str, bytes: &[u8], folder: Option<std::path::PathBuf>) -> Result<String, String> {
    let dir = folder.unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));
    let out = dir.join(filename);
    std::fs::write(&out, bytes).map_err(|e| format!("Save failed: {e}"))?;
    Ok(format!("Saved {}", crate::domain::paths::short_path(&out)))
}
