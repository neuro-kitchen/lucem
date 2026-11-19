//! Live 128 × 128 Visual Op Gallery with animated Before ↔ After hover transition.
//!
//! Reads a 128 × 128 center patch of the active image's active channel, computes fast CPU
//! previews for 14 curated operations across Filter / Restore / Threshold, and renders them
//! in a visual grid. Hovering any card smoothly animates a live Before ↔ After wipe across
//! the thumbnail so users can see the exact transformation before clicking to apply it.

use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage, SharedString, StatefulInteractiveElement as _, Styled as _, Task, Window,
};
use lucem_pipeline::Op;
use lucem_view::bgra_image;

use crate::data::store::Store;
use crate::domain::command::Command;
use crate::ui::theme::{self, color};

const THUMB: usize = 128;

pub type CloseGallery = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum GalleryCat {
    All,
    Filter,
    Restore,
    Threshold,
}

impl GalleryCat {
    const ALL: [Self; 4] = [Self::All, Self::Filter, Self::Restore, Self::Threshold];

    fn title(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Filter => "Filter",
            Self::Restore => "Restore & Enhance",
            Self::Threshold => "Threshold & Edges",
        }
    }
}

struct GalleryEntry {
    title: &'static str,
    blurb: &'static str,
    cat: GalleryCat,
    op: Op,
    after: Arc<RenderImage>,
}

pub struct OpGallery {
    store: Entity<Store>,
    close: CloseGallery,
    image_id: Option<String>,
    channel: usize,
    channel_name: String,
    cat: GalleryCat,
    before: Option<Arc<RenderImage>>,
    entries: Vec<GalleryEntry>,
    error: Option<String>,
    loading: bool,
    hovered: Option<usize>,
    /// 0.0..=1.0 smooth wipe position when hovering a card.
    wipe: f32,
    tick: f32,
    _anim: Option<Task<()>>,
}

impl OpGallery {
    pub fn new(store: Entity<Store>, close: CloseGallery, cx: &mut Context<Self>) -> Self {
        let s = store.read(cx);
        let active = s.doc.active().cloned();
        let py = s.active_pyramid().cloned();
        let (image_id, channel, channel_name, ch_window, ch_color, ch_gamma, z_cur) = match (&active, &py) {
            (Some(img), Some(py)) => {
                let c = img.channels.iter().position(|ch| ch.visible).unwrap_or(0);
                let ch = img.channels.get(c);
                let name = ch.map(|ch| ch.label.clone()).unwrap_or_else(|| format!("Channel {}", c + 1));
                let win = ch.map(|ch| (ch.min, ch.max)).unwrap_or((0.0, 1.0));
                let rgb = ch
                    .map(|ch| [ch.color[0] as f32 / 255.0, ch.color[1] as f32 / 255.0, ch.color[2] as f32 / 255.0])
                    .unwrap_or([1.0, 1.0, 1.0]);
                let gamma = ch.map(|ch| ch.gamma).unwrap_or(1.0);
                let z = s.cursor_of(&img.id).map(|c| c[0]).unwrap_or(0).min(py.depth().saturating_sub(1));
                (Some(img.id.clone()), c, name, win, rgb, gamma, z)
            }
            _ => (None, 0, String::new(), (0.0, 1.0), [1.0, 1.0, 1.0], 1.0, 0),
        };

        let mut this = Self {
            store,
            close,
            image_id: image_id.clone(),
            channel,
            channel_name,
            cat: GalleryCat::All,
            before: None,
            entries: Vec::new(),
            error: None,
            loading: py.is_some(),
            hovered: None,
            wipe: 1.0,
            tick: 0.0,
            _anim: None,
        };

        if let Some(py) = py {
            let work = cx.background_spawn(async move {
                // Choose a level where the smaller dimension is close to 128..512 so we see real structure.
                let level = py
                    .levels
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, l)| l.width.min(l.height) >= THUMB)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let l = &py.levels[level];
                let z_l = ((z_cur as f64 / l.z_downsample.max(1.0)).round() as usize).min(l.depth.saturating_sub(1));
                let cw = l.width.min(THUMB * 2);
                let ch_h = l.height.min(THUMB * 2);
                let x0 = l.width.saturating_sub(cw) / 2;
                let y0 = l.height.saturating_sub(ch_h) / 2;
                let raw = py.read_region_at(level, channel, z_l, x0, y0, cw, ch_h).map_err(|e| e.to_string())?;
                let patch = resample_normalize(&raw, cw, ch_h, THUMB, THUMB, ch_window, ch_gamma);
                let before_bgra = render_mono_bgra(&patch, ch_color);
                let specs = preview_specs();
                let mut results = Vec::with_capacity(specs.len());
                for (title, blurb, cat, op, kind) in specs {
                    let bgra = match kind {
                        PreviewKind::Image(f) => {
                            let out = f(&patch, THUMB, THUMB);
                            render_mono_bgra(&out, ch_color)
                        }
                        PreviewKind::Mask(f) => {
                            let mask = f(&patch, THUMB, THUMB);
                            render_mask_overlay_bgra(&patch, &mask, ch_color)
                        }
                    };
                    results.push((title, blurb, cat, op, bgra));
                }
                Ok::<_, String>((before_bgra, results))
            });
            cx.spawn(async move |this, cx| {
                let done = work.await;
                let _ = this.update(cx, |g, cx| {
                    g.loading = false;
                    match done {
                        Ok((before_bgra, results)) => {
                            g.before = Some(bgra_image(THUMB, THUMB, before_bgra));
                            g.entries = results
                                .into_iter()
                                .map(|(title, blurb, cat, op, bgra)| GalleryEntry {
                                    title,
                                    blurb,
                                    cat,
                                    op,
                                    after: bgra_image(THUMB, THUMB, bgra),
                                })
                                .collect();
                        }
                        Err(e) => g.error = Some(e),
                    }
                    cx.notify();
                });
            })
            .detach();
        } else {
            this.error = Some("Open an image first to preview operations.".into());
        }

        this
    }

    fn set_hovered(&mut self, idx: Option<usize>, cx: &mut Context<Self>) {
        if self.hovered == idx {
            return;
        }
        self.hovered = idx;
        if idx.is_some() {
            self.tick = 0.0;
            self.wipe = 0.5;
            self._anim = Some(cx.spawn(async move |this, cx| loop {
                cx.background_executor().timer(Duration::from_millis(40)).await;
                let keep = this
                    .update(cx, |g, cx| {
                        if g.hovered.is_none() {
                            return false;
                        }
                        g.tick += 0.04;
                        // Smooth 2.4 s back-and-forth Before <-> After sweep
                        let phase = (g.tick * std::f32::consts::TAU / 2.4).cos();
                        g.wipe = 0.5 + 0.46 * phase;
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);
                if !keep {
                    break;
                }
            }));
        } else {
            self._anim = None;
            self.wipe = 1.0;
        }
        cx.notify();
    }

    fn apply(&self, op: Op, window: &mut Window, cx: &mut Context<Self>) {
        let Some(image) = self.image_id.clone() else { return };
        let channel = self.channel;
        let title = op.title();
        self.store.update(cx, |s, cx| {
            let view = s.doc.graph.views().find(|v| s.doc.graph.images_of(&v.id).contains(&image)).map(|v| v.id.clone());
            if s.execute(Command::ApplyOp { op, view, image, channel }, cx).is_ok() {
                s.set_status(format!("{title} added to the pipeline"), cx);
            }
        });
        (self.close)(window, cx);
    }
}

impl Render for OpGallery {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (c1, c2) = (self.close.clone(), self.close.clone());
        let cat = self.cat;
        let before = self.before.clone();
        let hovered = self.hovered;
        let wipe = self.wipe;
        let me = cx.entity().downgrade();

        let visible_indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| cat == GalleryCat::All || e.cat == cat)
            .map(|(i, _)| i)
            .collect();

        div()
            .id("op-gallery-overlay")
            .absolute()
            .occlude()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.58))
            .on_click(move |_, w, cx| c1(w, cx))
            .child(
                v_flex()
                    .id("op-gallery-dialog")
                    .w(px(740.))
                    .max_h(px(640.))
                    .p(px(theme::LG))
                    .gap(px(theme::MD))
                    .rounded(px(theme::RADIUS_LG))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .items_center()
                            .child(Icon::new(IconName::LayoutGrid).small().text_color(color::accent_mark()))
                            .child(
                                v_flex()
                                    .flex_1()
                                    .min_w_0()
                                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Visual operation gallery"))
                                    .child(div().text_xs().text_color(color::text_muted()).child(format!(
                                        "Live 128 × 128 previews on {} · Hover any card for Before ↔ After transition · Click to apply",
                                        if self.channel_name.is_empty() { "your image" } else { &self.channel_name }
                                    ))),
                            )
                            .child(Button::new("gallery-close").ghost().xsmall().icon(Icon::new(IconName::Close)).on_click(move |_, w, cx| c2(w, cx))),
                    )
                    // Category filter pills
                    .child(
                        h_flex()
                            .gap(px(theme::XS))
                            .children(GalleryCat::ALL.into_iter().map(|c| {
                                let me = me.clone();
                                Button::new(SharedString::from(format!("gal-cat-{}", c.title())))
                                    .ghost()
                                    .xsmall()
                                    .selected(cat == c)
                                    .label(c.title())
                                    .on_click(move |_, _, cx| {
                                        let _ = me.update(cx, |g, cx| {
                                            g.cat = c;
                                            g.set_hovered(None, cx);
                                        });
                                    })
                            })),
                    )
                    .when(self.loading, |d| {
                        d.child(div().py(px(theme::XL)).text_sm().text_color(color::text_muted()).child("Computing live previews…"))
                    })
                    .children(self.error.clone().map(|e| div().py(px(theme::MD)).text_sm().text_color(color::danger()).child(e)))
                    .when(!self.loading && self.error.is_none(), |d| {
                        d.child(
                            div()
                                .id("op-gallery-grid")
                                .flex_1()
                                .min_h_0()
                                .max_h(px(470.))
                                .overflow_y_scroll()
                                .flex()
                                .flex_wrap()
                                .gap(px(theme::MD))
                                .children(visible_indices.into_iter().map(|idx| {
                                    let entry = &self.entries[idx];
                                    let is_hov = hovered == Some(idx);
                                    let split_px = if is_hov { (THUMB as f32 * wipe).clamp(4.0, THUMB as f32 - 4.0) } else { THUMB as f32 };
                                    let before_img = before.clone();
                                    let after_img = entry.after.clone();
                                    let op = entry.op.clone();
                                    let me_hov = me.clone();
                                    let me_click = me.clone();

                                    v_flex()
                                        .id(SharedString::from(format!("gal-card-{idx}")))
                                        .w(px(154.))
                                        .p(px(theme::SM))
                                        .gap(px(theme::XS))
                                        .rounded(px(theme::RADIUS_LG))
                                        .border_1()
                                        .border_color(if is_hov { color::accent_mark() } else { color::border() })
                                        .bg(color::panel())
                                        .cursor_pointer()
                                        .on_hover(move |enter, _, cx| {
                                            let enter = *enter;
                                            let _ = me_hov.update(cx, |g, cx| {
                                                if enter {
                                                    g.set_hovered(Some(idx), cx);
                                                } else if g.hovered == Some(idx) {
                                                    g.set_hovered(None, cx);
                                                }
                                            });
                                        })
                                        .on_click(move |_, window, cx| {
                                            let op = op.clone();
                                            let _ = me_click.update(cx, |g, cx| g.apply(op, window, cx));
                                        })
                                        .child(
                                            // 128x128 thumbnail with Before <-> After wipe transition on hover
                                            div()
                                                .relative()
                                                .size(px(THUMB as f32))
                                                .mx_auto()
                                                .rounded(px(4.))
                                                .overflow_hidden()
                                                .bg(gpui_kit::hsla(0., 0., 0., 1.0))
                                                .children(before_img.map(|b| img(b).size(px(THUMB as f32))))
                                                .child(
                                                    div()
                                                        .absolute()
                                                        .top_0()
                                                        .left_0()
                                                        .h(px(THUMB as f32))
                                                        .w(px(split_px))
                                                        .overflow_hidden()
                                                        .child(img(after_img).size(px(THUMB as f32))),
                                                )
                                                .when(is_hov, |box_div| {
                                                    box_div
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .top_0()
                                                                .left(px(split_px))
                                                                .w(px(1.5))
                                                                .h(px(THUMB as f32))
                                                                .bg(gpui_kit::hsla(0., 0., 1., 0.9)),
                                                        )
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .bottom(px(4.))
                                                                .left(px(4.))
                                                                .px(px(4.))
                                                                .py(px(1.))
                                                                .rounded(px(3.))
                                                                .bg(gpui_kit::hsla(0., 0., 0., 0.65))
                                                                .text_size(px(9.))
                                                                .text_color(gpui_kit::hsla(0., 0., 1., 0.95))
                                                                .child("After"),
                                                        )
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .bottom(px(4.))
                                                                .right(px(4.))
                                                                .px(px(4.))
                                                                .py(px(1.))
                                                                .rounded(px(3.))
                                                                .bg(gpui_kit::hsla(0., 0., 0., 0.65))
                                                                .text_size(px(9.))
                                                                .text_color(gpui_kit::hsla(0., 0., 1., 0.85))
                                                                .child("Before"),
                                                        )
                                                }),
                                        )
                                        .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(color::text()).truncate().child(entry.title))
                                        .child(div().text_size(px(10.)).text_color(color::text_muted()).truncate().child(entry.blurb))
                                })),
                        )
                    }),
            )
    }
}

// ----------------------------------------------------------------------------
// Fast pure-Rust 128 × 128 preview filters (sub-millisecond on CPU & WASM)
// ----------------------------------------------------------------------------

enum PreviewKind {
    Image(fn(&[f32], usize, usize) -> Vec<f32>),
    Mask(fn(&[f32], usize, usize) -> Vec<f32>),
}

fn preview_specs() -> Vec<(&'static str, &'static str, GalleryCat, Op, PreviewKind)> {
    vec![
        ("Gaussian blur", "Smooth sensor noise", GalleryCat::Filter, Op::GaussianBlur { sigma: 2.0 }, PreviewKind::Image(preview_gaussian)),
        ("Median", "Remove salt & pepper", GalleryCat::Filter, Op::Median { radius: 2 }, PreviewKind::Image(preview_median)),
        ("Unsharp mask", "Sharpen fine structures", GalleryCat::Filter, Op::UnsharpMask { sigma: 2.0, amount: 1.5 }, PreviewKind::Image(preview_unsharp)),
        ("Sobel edges", "Highlight boundaries", GalleryCat::Filter, Op::Sobel {}, PreviewKind::Image(preview_sobel)),
        ("Laplace", "Emphasize rapid edges", GalleryCat::Filter, Op::Laplace {}, PreviewKind::Image(preview_laplace)),
        ("Rolling ball", "Subtract uneven haze", GalleryCat::Restore, Op::RollingBall { radius: 16.0 }, PreviewKind::Image(preview_rolling_ball)),
        ("CLAHE", "Boost local contrast", GalleryCat::Restore, Op::Clahe { clip_limit: 2.0, tiles: 8 }, PreviewKind::Image(preview_clahe)),
        ("White top-hat", "Isolate bright spots", GalleryCat::Restore, Op::WhiteTophat { radius: 5 }, PreviewKind::Image(preview_tophat)),
        ("Invert", "Flip bright and dark", GalleryCat::Restore, Op::Invert {}, PreviewKind::Image(preview_invert)),
        ("Log contrast", "Lift dim structures", GalleryCat::Restore, Op::LogContrast { gain: 1.0 }, PreviewKind::Image(preview_log)),
        ("Otsu threshold", "Auto foreground mask", GalleryCat::Threshold, Op::OtsuThreshold { bins: 256 }, PreviewKind::Mask(preview_otsu)),
        ("Li threshold", "Low-contrast auto mask", GalleryCat::Threshold, Op::LiThreshold { bins: 256 }, PreviewKind::Mask(preview_li)),
        ("Triangle", "Sparse-object mask", GalleryCat::Threshold, Op::TriangleThreshold { bins: 256 }, PreviewKind::Mask(preview_triangle)),
        ("Canny edges", "Crisp edge contours", GalleryCat::Threshold, Op::Canny { sigma: 1.4 }, PreviewKind::Mask(preview_canny)),
    ]
}

fn resample_normalize(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize, window: (f32, f32), gamma: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * dh];
    if sw == 0 || sh == 0 || src.is_empty() {
        return out;
    }
    let (lo, hi) = if (window.1 - window.0).abs() > 1e-9 {
        window
    } else {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in src {
            if v.is_finite() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if mx > mn { (mn, mx) } else { (0.0, 1.0) }
    };
    let span = (hi - lo).max(1e-9);
    let g = gamma.clamp(0.1, 5.0);
    for y in 0..dh {
        let sy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = sy - y0 as f32;
        for x in 0..dw {
            let sx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = sx - x0 as f32;
            let v00 = src[y0 * sw + x0];
            let v10 = src[y0 * sw + x1];
            let v01 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            let v = (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy) + (v01 * (1.0 - fx) + v11 * fx) * fy;
            let n = ((v - lo) / span).clamp(0.0, 1.0);
            out[y * dw + x] = if (g - 1.0).abs() < 1e-3 { n } else { n.powf(g) };
        }
    }
    out
}

fn render_mono_bgra(img: &[f32], rgb: [f32; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(img.len() * 4);
    for &v in img {
        let v = v.clamp(0.0, 1.0);
        out.push((v * rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((v * rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((v * rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push(255);
    }
    out
}

fn render_mask_overlay_bgra(base: &[f32], mask: &[f32], rgb: [f32; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity(base.len() * 4);
    for (&v, &m) in base.iter().zip(mask.iter()) {
        let v = v.clamp(0.0, 1.0);
        let mut r = v * rgb[0];
        let mut g = v * rgb[1];
        let mut b = v * rgb[2];
        if m > 0.5 {
            // Crisp cyan tint over masked pixels so segmentation is immediately legible
            r = r * 0.45 + 0.10;
            g = g * 0.45 + 0.85;
            b = b * 0.45 + 0.95;
        }
        out.push((b * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((g * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push((r * 255.0).round().clamp(0.0, 255.0) as u8);
        out.push(255);
    }
    out
}

fn blur_separable(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let r = (sigma * 2.5).ceil() as isize;
    let mut k = Vec::with_capacity((2 * r + 1) as usize);
    let mut sum = 0.0f32;
    for d in -r..=r {
        let wt = (-0.5 * (d as f32 / sigma).powi(2)).exp();
        k.push(wt);
        sum += wt;
    }
    for wt in &mut k {
        *wt /= sum;
    }
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &wt) in k.iter().enumerate() {
                let sx = (x as isize + i as isize - r).clamp(0, w as isize - 1) as usize;
                acc += src[row + sx] * wt;
            }
            tmp[row + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (i, &wt) in k.iter().enumerate() {
                let sy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * wt;
            }
            out[y * w + x] = acc;
        }
    }
    out
}

fn preview_gaussian(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    blur_separable(src, w, h, 2.2)
}

fn preview_median(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let mut buf = [0.0f32; 25];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for dy in -2isize..=2 {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for dx in -2isize..=2 {
                    let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    buf[n] = src[sy * w + sx];
                    n += 1;
                }
            }
            buf.sort_unstable_by(|a, b| a.total_cmp(b));
            out[y * w + x] = buf[12];
        }
    }
    out
}

fn preview_unsharp(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let blurred = blur_separable(src, w, h, 2.0);
    src.iter().zip(blurred.iter()).map(|(&a, &b)| (a + 1.6 * (a - b)).clamp(0.0, 1.0)).collect()
}

fn preview_sobel(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let at = |x: isize, y: isize| src[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    let mut mx = 1e-6f32;
    for y in 0..h as isize {
        for x in 0..w as isize {
            let gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y) + 2.0 * at(x + 1, y) - at(x - 1, y + 1) + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1);
            let m = gx.hypot(gy);
            mx = mx.max(m);
            out[y as usize * w + x as usize] = m;
        }
    }
    for v in &mut out {
        *v = (*v / mx).clamp(0.0, 1.0);
    }
    out
}

fn preview_laplace(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let at = |x: isize, y: isize| src[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    let mut mx = 1e-6f32;
    for y in 0..h as isize {
        for x in 0..w as isize {
            let v = (4.0 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)).abs();
            mx = mx.max(v);
            out[y as usize * w + x as usize] = v;
        }
    }
    for v in &mut out {
        *v = (*v / mx).clamp(0.0, 1.0);
    }
    out
}

fn morph_erode(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut m = f32::INFINITY;
            for dx in -r..=r {
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                m = m.min(src[row + sx]);
            }
            tmp[row + x] = m;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::INFINITY;
            for dy in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                m = m.min(tmp[sy * w + x]);
            }
            out[y * w + x] = m;
        }
    }
    out
}

fn morph_dilate(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for dx in -r..=r {
                let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                m = m.max(src[row + sx]);
            }
            tmp[row + x] = m;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut m = f32::NEG_INFINITY;
            for dy in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                m = m.max(tmp[sy * w + x]);
            }
            out[y * w + x] = m;
        }
    }
    out
}

fn preview_rolling_ball(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let bg = morph_dilate(&morph_erode(src, w, h, 8), w, h, 8);
    let bg = blur_separable(&bg, w, h, 3.0);
    let mut out: Vec<f32> = src.iter().zip(bg.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect();
    let mx = out.iter().copied().fold(1e-6f32, f32::max);
    for v in &mut out {
        *v = (*v / mx).clamp(0.0, 1.0);
    }
    out
}

fn preview_tophat(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let opened = morph_dilate(&morph_erode(src, w, h, 4), w, h, 4);
    let mut out: Vec<f32> = src.iter().zip(opened.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect();
    let mx = out.iter().copied().fold(1e-6f32, f32::max);
    for v in &mut out {
        *v = (*v / mx).clamp(0.0, 1.0);
    }
    out
}

fn preview_clahe(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Local contrast normalization + mild histogram equalization
    let mean = blur_separable(src, w, h, 8.0);
    let sq: Vec<f32> = src.iter().zip(mean.iter()).map(|(&a, &m)| (a - m) * (a - m)).collect();
    let var = blur_separable(&sq, w, h, 8.0);
    src.iter()
        .zip(mean.iter().zip(var.iter()))
        .map(|(&a, (&m, &v))| {
            let std = v.sqrt().max(0.06);
            (0.5 + (a - m) / (2.4 * std)).clamp(0.0, 1.0)
        })
        .collect()
}

fn preview_invert(src: &[f32], _: usize, _: usize) -> Vec<f32> {
    src.iter().map(|&v| 1.0 - v.clamp(0.0, 1.0)).collect()
}

fn preview_log(src: &[f32], _: usize, _: usize) -> Vec<f32> {
    let denom = 10.0f32.ln();
    src.iter().map(|&v| ((1.0 + v.clamp(0.0, 1.0) * 9.0).ln() / denom).clamp(0.0, 1.0)).collect()
}

fn histogram_256(src: &[f32]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &v in src {
        let bin = (v.clamp(0.0, 1.0) * 255.0).round() as usize;
        h[bin.min(255)] += 1;
    }
    h
}

fn preview_otsu(src: &[f32], _: usize, _: usize) -> Vec<f32> {
    let h = histogram_256(src);
    let total = src.len() as f64;
    let sum_all: f64 = h.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w_b, mut sum_b, mut best_var, mut best_t) = (0.0, 0.0, -1.0, 128usize);
    for (t, &c) in h.iter().enumerate() {
        w_b += c as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * c as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t;
        }
    }
    let thresh = best_t as f32 / 255.0;
    src.iter().map(|&v| if v > thresh { 1.0 } else { 0.0 }).collect()
}

fn preview_li(src: &[f32], _: usize, _: usize) -> Vec<f32> {
    // Iterative minimum cross-entropy (Li)
    let mut t = src.iter().sum::<f32>() / (src.len().max(1) as f32);
    for _ in 0..20 {
        let (mut s_lo, mut n_lo, mut s_hi, mut n_hi) = (0.0f32, 0usize, 0.0f32, 0usize);
        for &v in src {
            if v <= t {
                s_lo += v;
                n_lo += 1;
            } else {
                s_hi += v;
                n_hi += 1;
            }
        }
        if n_lo == 0 || n_hi == 0 {
            break;
        }
        let m_lo = (s_lo / n_lo as f32).max(1e-4);
        let m_hi = (s_hi / n_hi as f32).max(1e-4);
        let next = (m_hi - m_lo) / (m_hi.ln() - m_lo.ln()).max(1e-4);
        if (next - t).abs() < 1e-3 {
            t = next;
            break;
        }
        t = next;
    }
    src.iter().map(|&v| if v > t { 1.0 } else { 0.0 }).collect()
}

fn preview_triangle(src: &[f32], _: usize, _: usize) -> Vec<f32> {
    let h = histogram_256(src);
    let (peak_i, &peak_h) = h.iter().enumerate().max_by_key(|(_, c)| *c).unwrap_or((0, &1));
    let last_i = h.iter().rposition(|&c| c > 0).unwrap_or(255).max(peak_i + 1);
    let (dx, dy) = ((last_i - peak_i) as f32, -(peak_h as f32));
    let len = dx.hypot(dy).max(1.0);
    let mut best_i = peak_i;
    let mut best_d = -1.0f32;
    for i in peak_i..=last_i {
        let d = ((i - peak_i) as f32 * dy - (h[i] as f32 - peak_h as f32) * dx).abs() / len;
        if d > best_d {
            best_d = d;
            best_i = i;
        }
    }
    let thresh = best_i as f32 / 255.0;
    src.iter().map(|&v| if v > thresh { 1.0 } else { 0.0 }).collect()
}

fn preview_canny(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let smooth = blur_separable(src, w, h, 1.2);
    let grad = preview_sobel(&smooth, w, h);
    grad.into_iter().map(|v| if v > 0.22 { 1.0 } else { 0.0 }).collect()
}
