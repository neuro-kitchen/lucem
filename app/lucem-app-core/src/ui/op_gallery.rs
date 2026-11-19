//! Instagram Edits-style non-blocking bottom Visual Effect Strip (`OpGallery`) and floating
//! animated Before ↔ After hover preview card (`OpHoverPreview`).
//!
//! Both reuse fast sub-millisecond pure-Rust 128 × 128 CPU preview filters so previews work
//! identically on desktop and in the browser (WASM) with zero GPU round-trip latency.

use std::sync::Arc;
use std::time::Duration;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, img, px, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, RenderImage, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Task, Window,
};
use lucem_pipeline::Op;
use lucem_view::bgra_image;

use crate::data::store::{AppEvent, Store};
use crate::domain::command::Command;
use crate::ui::theme::{self, color};
use crate::ui::widgets::stepper;

const THUMB: usize = 128;

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
            Self::Restore => "Restore",
            Self::Threshold => "Threshold",
        }
    }
}

struct StripEntry {
    title: &'static str,
    blurb: &'static str,
    cat: GalleryCat,
    op: Op,
    after: Arc<RenderImage>,
}

/// Non-blocking bottom Visual Effect Strip (Instagram Edits style) floated over the Explore
/// canvas so the full microscopy image remains visible above while browsing & tuning ops.
pub struct OpGallery {
    store: Entity<Store>,
    image_key: Option<(String, usize, usize)>,
    image_id: Option<String>,
    channel: usize,
    channel_name: String,
    ch_color: [f32; 3],
    cat: GalleryCat,
    patch: Option<Arc<Vec<f32>>>,
    before: Option<Arc<RenderImage>>,
    entries: Vec<StripEntry>,
    selected: Option<usize>,
    selected_op: Option<Op>,
    selected_preview: Option<Arc<RenderImage>>,
    hovered: Option<usize>,
    wipe: f32,
    tick: f32,
    loading: bool,
    error: Option<String>,
    _anim: Option<Task<()>>,
    _subs: Vec<Subscription>,
}

impl OpGallery {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let subs = vec![cx.subscribe(&store, |this, _, e: &AppEvent, cx| {
            if matches!(e, AppEvent::Document(_) | AppEvent::ImageReady(_) | AppEvent::Cursor) {
                this.refresh_if_needed(cx);
            }
        })];
        let mut this = Self {
            store,
            image_key: None,
            image_id: None,
            channel: 0,
            channel_name: String::new(),
            ch_color: [1.0, 1.0, 1.0],
            cat: GalleryCat::All,
            patch: None,
            before: None,
            entries: Vec::new(),
            selected: None,
            selected_op: None,
            selected_preview: None,
            hovered: None,
            wipe: 1.0,
            tick: 0.0,
            loading: false,
            error: None,
            _anim: None,
            _subs: subs,
        };
        this.refresh_if_needed(cx);
        this
    }

    pub fn refresh_if_needed(&mut self, cx: &mut Context<Self>) {
        let s = self.store.read(cx);
        if !s.visual_strip_open {
            return;
        }
        let (Some(img), Some(py)) = (s.doc.active().cloned(), s.active_pyramid().cloned()) else {
            self.error = Some("Open an image first to preview operations.".into());
            self.loading = false;
            return;
        };
        let c = img.channels.iter().position(|ch| ch.visible).unwrap_or(0);
        let z = s.cursor_of(&img.id).map(|cur| cur[0]).unwrap_or(0).min(py.depth().saturating_sub(1));
        let key = (img.id.clone(), c, z);
        if self.image_key.as_ref() == Some(&key) && !self.entries.is_empty() {
            return;
        }
        let ch = img.channels.get(c);
        let name = ch.map(|ch| ch.label.clone()).unwrap_or_else(|| format!("Channel {}", c + 1));
        let win = ch.map(|ch| (ch.min, ch.max)).unwrap_or((0.0, 1.0));
        let rgb = ch
            .map(|ch| [ch.color[0] as f32 / 255.0, ch.color[1] as f32 / 255.0, ch.color[2] as f32 / 255.0])
            .unwrap_or([1.0, 1.0, 1.0]);
        let gamma = ch.map(|ch| ch.gamma).unwrap_or(1.0);

        self.image_key = Some(key);
        self.image_id = Some(img.id.clone());
        self.channel = c;
        self.channel_name = name;
        self.ch_color = rgb;
        self.loading = true;
        self.error = None;

        let work = cx.background_spawn(async move {
            let patch = read_center_patch(&py, c, z, win, gamma)?;
            let before_bgra = render_mono_bgra(&patch, rgb);
            let specs = strip_specs();
            let mut results = Vec::with_capacity(specs.len());
            for (title, blurb, cat, op) in specs {
                let bgra = render_op_preview_bgra(&patch, THUMB, THUMB, &op, rgb);
                results.push((title, blurb, cat, op, bgra));
            }
            Ok::<_, String>((patch, before_bgra, results))
        });
        cx.spawn(async move |this, cx| {
            let done = work.await;
            let _ = this.update(cx, |g, cx| {
                g.loading = false;
                match done {
                    Ok((patch, before_bgra, results)) => {
                        g.patch = Some(Arc::new(patch));
                        g.before = Some(bgra_image(THUMB, THUMB, before_bgra));
                        g.entries = results
                            .into_iter()
                            .map(|(title, blurb, cat, op, bgra)| StripEntry {
                                title,
                                blurb,
                                cat,
                                op,
                                after: bgra_image(THUMB, THUMB, bgra),
                            })
                            .collect();
                        g.selected = None;
                        g.selected_op = None;
                        g.selected_preview = None;
                    }
                    Err(e) => g.error = Some(e),
                }
                cx.notify();
            });
        })
        .detach();
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
                        let phase = (g.tick * std::f32::consts::TAU / 2.2).cos();
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

    fn select(&mut self, idx: usize, cx: &mut Context<Self>) {
        if self.selected == Some(idx) {
            if let Some(op) = self.selected_op.clone() {
                self.apply(op, cx);
            }
            return;
        }
        if let Some(entry) = self.entries.get(idx) {
            self.selected = Some(idx);
            self.selected_op = Some(entry.op.clone());
            self.selected_preview = Some(entry.after.clone());
            cx.notify();
        }
    }

    fn nudge_selected_param(&mut self, delta: i32, cx: &mut Context<Self>) {
        let Some(op) = self.selected_op.as_mut() else { return };
        if !step_primary_param(op, delta) {
            return;
        }
        if let Some(patch) = self.patch.as_ref() {
            let bgra = render_op_preview_bgra(patch, THUMB, THUMB, op, self.ch_color);
            let img = bgra_image(THUMB, THUMB, bgra);
            self.selected_preview = Some(img.clone());
            if let Some(idx) = self.selected
                && let Some(e) = self.entries.get_mut(idx)
            {
                e.after = img;
                e.op = op.clone();
            }
        }
        cx.notify();
    }

    fn apply(&mut self, op: Op, cx: &mut Context<Self>) {
        let Some(image) = self.image_id.clone() else { return };
        let channel = self.channel;
        let title = op.title();
        self.store.update(cx, |s, cx| {
            let view = s.doc.graph.views().find(|v| s.doc.graph.images_of(&v.id).contains(&image)).map(|v| v.id.clone());
            if s.execute(Command::ApplyOp { op, view, image, channel }, cx).is_ok() {
                s.set_status(format!("{title} added to the pipeline"), cx);
            }
            s.set_visual_strip(false, cx);
        });
    }
}

impl Render for OpGallery {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.refresh_if_needed(cx);
        let cat = self.cat;
        let before = self.before.clone();
        let hovered = self.hovered;
        let selected = self.selected;
        let wipe = self.wipe;
        let me = cx.entity().downgrade();
        let st_close = self.store.clone();

        let visible_indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| cat == GalleryCat::All || e.cat == cat)
            .map(|(i, _)| i)
            .collect();

        // Inline primary parameter bar when an op is selected (Instagram Edits style)
        let tune_bar = self.selected_op.clone().map(|op| {
            let me_step = me.clone();
            let me_apply = me.clone();
            let param_label = primary_param_label(&op);
            let title = op.title();
            let op_apply = op.clone();
            h_flex()
                .px(px(theme::MD))
                .py(px(theme::XS))
                .gap(px(theme::SM))
                .items_center()
                .rounded(px(999.))
                .border_1()
                .border_color(color::border())
                .bg(color::raised())
                .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).child(title))
                .children(param_label.map(|(name, val)| {
                    h_flex()
                        .gap(px(theme::XS))
                        .items_center()
                        .child(div().text_xs().text_color(color::text_muted()).child(name))
                        .child(stepper("strip-param", val, move |d, _, cx| {
                            let step = if d >= 0.0 { 1 } else { -1 };
                            let _ = me_step.update(cx, |g, cx| g.nudge_selected_param(step, cx));
                        }))
                }))
                .child(
                    Button::new("strip-apply")
                        .primary()
                        .xsmall()
                        .label("Apply")
                        .on_click(move |_, _, cx| {
                            let op = op_apply.clone();
                            let _ = me_apply.update(cx, |g, cx| g.apply(op, cx));
                        }),
                )
        });

        v_flex()
            .id("visual-effect-strip")
            .occlude()
            .absolute()
            .bottom(px(14.))
            .left(px(56.))
            .right(px(16.))
            .gap(px(theme::SM))
            .items_center()
            // Floating inline parameter scrubber pill above the filmstrip when an op is selected
            .children(tune_bar)
            // Main M3 Expressive floating pill tray
            .child(
                v_flex()
                    .w_full()
                    .p(px(theme::MD))
                    .gap(px(theme::SM))
                    .rounded(px(18.))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .child(
                        h_flex()
                            .gap(px(theme::SM))
                            .items_center()
                            .child(Icon::new(IconName::Sparkles).small().text_color(color::accent_mark()))
                            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).child("Visual effects"))
                            .child(div().text_xs().text_color(color::text_muted()).truncate().child(format!(
                                "· {} · Hover for Before ↔ After · Click to tune, click again to apply",
                                if self.channel_name.is_empty() { "Active channel" } else { &self.channel_name }
                            )))
                            .child(div().flex_1())
                            .children(GalleryCat::ALL.into_iter().map(|c| {
                                let me = me.clone();
                                Button::new(SharedString::from(format!("strip-cat-{}", c.title())))
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
                            }))
                            .child(
                                Button::new("strip-close")
                                    .ghost()
                                    .xsmall()
                                    .icon(Icon::new(IconName::Close))
                                    .tooltip("Close visual strip")
                                    .on_click(move |_, _, cx| st_close.update(cx, |s, cx| s.set_visual_strip(false, cx))),
                            ),
                    )
                    .when(self.loading, |d| d.child(div().py(px(theme::MD)).text_xs().text_color(color::text_muted()).child("Computing live previews…")))
                    .children(self.error.clone().map(|e| div().py(px(theme::SM)).text_xs().text_color(color::danger()).child(e)))
                    .when(!self.loading && self.error.is_none(), |d| {
                        d.child(
                            h_flex()
                                .id("visual-strip-scroll")
                                .w_full()
                                .overflow_x_scroll()
                                .gap(px(theme::SM))
                                .pb(px(2.))
                                .children(visible_indices.into_iter().map(|idx| {
                                    let entry = &self.entries[idx];
                                    let is_hov = hovered == Some(idx);
                                    let is_sel = selected == Some(idx);
                                    let thumb_px = 92.0f32;
                                    let split_px = if is_hov { (thumb_px * wipe).clamp(4.0, thumb_px - 4.0) } else { thumb_px };
                                    let before_img = before.clone();
                                    let after_img = entry.after.clone();
                                    let me_hov = me.clone();
                                    let me_click = me.clone();

                                    v_flex()
                                        .id(SharedString::from(format!("strip-card-{idx}")))
                                        .flex_none()
                                        .w(px(110.))
                                        .p(px(6.))
                                        .gap(px(4.))
                                        .rounded(px(12.))
                                        .border_1()
                                        .border_color(if is_sel || is_hov { color::accent_mark() } else { color::border() })
                                        .bg(if is_sel { color::control() } else { color::panel() })
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
                                        .on_click(move |_, _, cx| {
                                            let _ = me_click.update(cx, |g, cx| g.select(idx, cx));
                                        })
                                        .child(
                                            div()
                                                .relative()
                                                .size(px(thumb_px))
                                                .mx_auto()
                                                .rounded(px(8.))
                                                .overflow_hidden()
                                                .bg(gpui_kit::hsla(0., 0., 0., 1.0))
                                                .children(before_img.map(|b| img(b).size(px(thumb_px))))
                                                .child(
                                                    div()
                                                        .absolute()
                                                        .top_0()
                                                        .left_0()
                                                        .h(px(thumb_px))
                                                        .w(px(split_px))
                                                        .overflow_hidden()
                                                        .child(img(after_img).size(px(thumb_px))),
                                                )
                                                .when(is_hov, |box_div| {
                                                    box_div
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .top_0()
                                                                .left(px(split_px))
                                                                .w(px(1.5))
                                                                .h(px(thumb_px))
                                                                .bg(gpui_kit::hsla(0., 0., 1., 0.9)),
                                                        )
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .top(px(3.))
                                                                .left(px(3.))
                                                                .px(px(3.))
                                                                .py(px(1.))
                                                                .rounded(px(3.))
                                                                .bg(gpui_kit::hsla(0., 0., 0., 0.68))
                                                                .text_size(px(8.))
                                                                .text_color(gpui_kit::hsla(0., 0., 1., 0.95))
                                                                .child("After"),
                                                        )
                                                        .child(
                                                            div()
                                                                .absolute()
                                                                .top(px(3.))
                                                                .right(px(3.))
                                                                .px(px(3.))
                                                                .py(px(1.))
                                                                .rounded(px(3.))
                                                                .bg(gpui_kit::hsla(0., 0., 0., 0.68))
                                                                .text_size(px(8.))
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
// Floating Before <-> After Hover Preview Card (for any hovered op anywhere)
// ----------------------------------------------------------------------------

/// Floating animated Before ↔ After preview card driven by `Store::hovered_op`.
/// Appears whenever the user hovers an operation chip or quick-pick op, showing a looping
/// pre ↔ post wipe on the active image (or a crisp synthetic microscopy patch if none is open).
pub struct OpHoverPreview {
    store: Entity<Store>,
    cached_key: Option<(Option<String>, usize, usize)>,
    cached_patch: Option<(Vec<f32>, [f32; 3], Arc<RenderImage>)>,
    last_op: Option<Op>,
    after_img: Option<Arc<RenderImage>>,
    wipe: f32,
    tick: f32,
    _anim: Option<Task<()>>,
    _sub: Subscription,
}

impl OpHoverPreview {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let sub = cx.observe(&store, |this, _, cx| {
            this.sync_hover(cx);
        });
        Self {
            store,
            cached_key: None,
            cached_patch: None,
            last_op: None,
            after_img: None,
            wipe: 0.5,
            tick: 0.0,
            _anim: None,
            _sub: sub,
        }
    }

    fn sync_hover(&mut self, cx: &mut Context<Self>) {
        let op = self.store.read(cx).hovered_op.clone();
        if op == self.last_op {
            return;
        }
        self.last_op = op.clone();
        if let Some(op) = op {
            // Ensure we have a 128x128 base patch (from active image or synthetic fallback)
            let s = self.store.read(cx);
            let (key, needs_read) = match (s.doc.active(), s.active_pyramid()) {
                (Some(img), Some(py)) => {
                    let c = img.channels.iter().position(|ch| ch.visible).unwrap_or(0);
                    let z = s.cursor_of(&img.id).map(|cur| cur[0]).unwrap_or(0).min(py.depth().saturating_sub(1));
                    let k = (Some(img.id.clone()), c, z);
                    let miss = self.cached_key.as_ref() != Some(&k);
                    (k, miss)
                }
                _ => {
                    let k = (None, 0, 0);
                    let miss = self.cached_key.as_ref() != Some(&k);
                    (k, miss)
                }
            };
            if needs_read {
                let (patch, rgb) = match (s.doc.active(), s.active_pyramid()) {
                    (Some(img), Some(py)) => {
                        let c = key.1;
                        let ch = img.channels.get(c);
                        let win = ch.map(|ch| (ch.min, ch.max)).unwrap_or((0.0, 1.0));
                        let rgb = ch
                            .map(|ch| [ch.color[0] as f32 / 255.0, ch.color[1] as f32 / 255.0, ch.color[2] as f32 / 255.0])
                            .unwrap_or([1.0, 1.0, 1.0]);
                        let gamma = ch.map(|ch| ch.gamma).unwrap_or(1.0);
                        match read_center_patch(py, c, key.2, win, gamma) {
                            Ok(p) => (p, rgb),
                            Err(_) => (synthetic_cells_patch(THUMB, THUMB), [0.35, 0.85, 0.95]),
                        }
                    }
                    _ => (synthetic_cells_patch(THUMB, THUMB), [0.35, 0.85, 0.95]),
                };
                let before = bgra_image(THUMB, THUMB, render_mono_bgra(&patch, rgb));
                self.cached_key = Some(key);
                self.cached_patch = Some((patch, rgb, before));
            }
            if let Some((patch, rgb, _)) = self.cached_patch.as_ref() {
                let after_bgra = render_op_preview_bgra(patch, THUMB, THUMB, &op, *rgb);
                self.after_img = Some(bgra_image(THUMB, THUMB, after_bgra));
            }
            self.tick = 0.0;
            self.wipe = 0.5;
            self._anim = Some(cx.spawn(async move |this, cx| loop {
                cx.background_executor().timer(Duration::from_millis(40)).await;
                let keep = this
                    .update(cx, |p, cx| {
                        if p.last_op.is_none() {
                            return false;
                        }
                        p.tick += 0.04;
                        let phase = (p.tick * std::f32::consts::TAU / 2.2).cos();
                        p.wipe = 0.5 + 0.46 * phase;
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
            self.after_img = None;
        }
        cx.notify();
    }
}

impl Render for OpHoverPreview {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let Some(op) = self.last_op.as_ref() else {
            return div().into_any_element();
        };
        let Some((_, _, before)) = self.cached_patch.as_ref() else {
            return div().into_any_element();
        };
        let Some(after) = self.after_img.as_ref() else {
            return div().into_any_element();
        };
        let box_px = 156.0f32;
        let split_px = (box_px * self.wipe).clamp(4.0, box_px - 4.0);
        v_flex()
            .id("op-hover-preview-card")
            .absolute()
            .bottom(px(44.))
            .right(px(340.))
            .w(px(184.))
            .p(px(theme::SM))
            .gap(px(theme::XS))
            .rounded(px(16.))
            .border_1()
            .border_color(color::accent_mark())
            .bg(color::raised())
            .child(
                div()
                    .relative()
                    .size(px(box_px))
                    .mx_auto()
                    .rounded(px(10.))
                    .overflow_hidden()
                    .bg(gpui_kit::hsla(0., 0., 0., 1.0))
                    .child(img(before.clone()).size(px(box_px)))
                    .child(
                        div()
                            .absolute()
                            .top_0()
                            .left_0()
                            .h(px(box_px))
                            .w(px(split_px))
                            .overflow_hidden()
                            .child(img(after.clone()).size(px(box_px))),
                    )
                    .child(
                        div()
                            .absolute()
                            .top_0()
                            .left(px(split_px))
                            .w(px(1.5))
                            .h(px(box_px))
                            .bg(gpui_kit::hsla(0., 0., 1., 0.92)),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(4.))
                            .left(px(4.))
                            .px(px(5.))
                            .py(px(1.))
                            .rounded(px(999.))
                            .bg(gpui_kit::hsla(0., 0., 0., 0.68))
                            .text_size(px(9.))
                            .text_color(gpui_kit::hsla(0., 0., 1., 0.95))
                            .child("After"),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(px(4.))
                            .right(px(4.))
                            .px(px(5.))
                            .py(px(1.))
                            .rounded(px(999.))
                            .bg(gpui_kit::hsla(0., 0., 0., 0.68))
                            .text_size(px(9.))
                            .text_color(gpui_kit::hsla(0., 0., 1., 0.85))
                            .child("Before"),
                    ),
            )
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text()).truncate().child(op.title()))
            .child(div().text_size(px(10.)).text_color(color::text_muted()).child(op.description()))
            .into_any_element()
    }
}

/// Builds a `PopupMenuItem` for an operation that drives the floating Before ↔ After hover
/// preview card whenever the user hovers over the menu row.
pub fn op_menu_item(
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    icon: IconName,
    op: Op,
    store: Entity<Store>,
    on_click: impl Fn(&mut Window, &mut gpui_kit::App) + 'static,
) -> gpui_kit::component::menu::PopupMenuItem {
    let id = id.into();
    let label = label.into();
    let op_hov = op.clone();
    let st_hov = store.clone();
    let st_click = store.clone();
    gpui_kit::component::menu::PopupMenuItem::element(move |_, _| {
        let op_hov = op_hov.clone();
        let st_hov = st_hov.clone();
        h_flex()
            .id(id.clone())
            .w_full()
            .gap(px(8.))
            .items_center()
            .child(Icon::new(icon).small().text_color(color::text_muted()))
            .child(div().text_sm().child(label.clone()))
            .on_hover(move |enter, _, cx| {
                let next = (*enter).then(|| op_hov.clone());
                st_hov.update(cx, |s, cx| s.set_hovered_op(next, cx));
            })
    })
    .on_click(move |_, w, cx| {
        st_click.update(cx, |s, cx| s.set_hovered_op(None, cx));
        on_click(w, cx);
    })
}

// ----------------------------------------------------------------------------
// Specs, primary parameter tuning & pure-Rust 128 × 128 preview filters
// ----------------------------------------------------------------------------

fn strip_specs() -> Vec<(&'static str, &'static str, GalleryCat, Op)> {
    vec![
        ("Gaussian blur", "Smooth sensor noise", GalleryCat::Filter, Op::GaussianBlur { sigma: 2.0 }),
        ("Median", "Remove hot pixels", GalleryCat::Filter, Op::Median { radius: 2 }),
        ("Unsharp mask", "Sharpen structures", GalleryCat::Filter, Op::UnsharpMask { sigma: 2.0, amount: 1.5 }),
        ("Sobel edges", "Highlight boundaries", GalleryCat::Filter, Op::Sobel {}),
        ("Laplace", "Emphasize rapid edges", GalleryCat::Filter, Op::Laplace {}),
        ("Rolling ball", "Subtract uneven haze", GalleryCat::Restore, Op::RollingBall { radius: 16.0 }),
        ("CLAHE", "Boost local contrast", GalleryCat::Restore, Op::Clahe { clip_limit: 2.0, tiles: 8 }),
        ("White top-hat", "Isolate bright spots", GalleryCat::Restore, Op::WhiteTophat { radius: 5 }),
        ("Invert", "Flip bright and dark", GalleryCat::Restore, Op::Invert {}),
        ("Log contrast", "Lift dim structures", GalleryCat::Restore, Op::LogContrast { gain: 1.0 }),
        ("Otsu threshold", "Auto foreground mask", GalleryCat::Threshold, Op::OtsuThreshold { bins: 256 }),
        ("Li threshold", "Dim-signal auto mask", GalleryCat::Threshold, Op::LiThreshold { bins: 256 }),
        ("Triangle", "Sparse-object mask", GalleryCat::Threshold, Op::TriangleThreshold { bins: 256 }),
        ("Canny edges", "Crisp edge contours", GalleryCat::Threshold, Op::Canny { sigma: 1.4 }),
    ]
}

fn primary_param_label(op: &Op) -> Option<(&'static str, String)> {
    match op {
        Op::GaussianBlur { sigma } | Op::Canny { sigma } => Some(("Sigma", format!("{sigma:.1} px"))),
        Op::UnsharpMask { amount, .. } => Some(("Amount", format!("{amount:.1}×"))),
        Op::Median { radius } | Op::WhiteTophat { radius } | Op::BlackTophat { radius } => Some(("Radius", format!("{radius} px"))),
        Op::RollingBall { radius } => Some(("Radius", format!("{radius:.0} px"))),
        Op::Clahe { clip_limit, .. } => Some(("Clip", format!("{clip_limit:.1}"))),
        Op::LogContrast { gain } => Some(("Gain", format!("{gain:.1}"))),
        _ => None,
    }
}

fn step_primary_param(op: &mut Op, delta: i32) -> bool {
    let d = delta as f64;
    match op {
        Op::GaussianBlur { sigma } | Op::Canny { sigma } => {
            *sigma = (*sigma + d * 0.5).clamp(0.5, 12.0);
            true
        }
        Op::UnsharpMask { amount, .. } => {
            *amount = (*amount + d * 0.3).clamp(0.3, 5.0);
            true
        }
        Op::Median { radius } | Op::WhiteTophat { radius } | Op::BlackTophat { radius } => {
            *radius = (*radius as isize + delta as isize).clamp(1, 12) as usize;
            true
        }
        Op::RollingBall { radius } => {
            *radius = (*radius + d * 4.0).clamp(4.0, 64.0);
            true
        }
        Op::Clahe { clip_limit, .. } => {
            *clip_limit = (*clip_limit + d * 0.5).clamp(0.5, 8.0);
            true
        }
        Op::LogContrast { gain } => {
            *gain = (*gain + d * 0.5).clamp(0.5, 5.0);
            true
        }
        _ => false,
    }
}

fn read_center_patch(py: &lucem_io::Pyramid, channel: usize, z_cur: usize, window: (f32, f32), gamma: f32) -> Result<Vec<f32>, String> {
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
    Ok(resample_normalize(&raw, cw, ch_h, THUMB, THUMB, window, gamma))
}

fn synthetic_cells_patch(w: usize, h: usize) -> Vec<f32> {
    let centers = [(34.0f32, 38.0, 14.0), (86.0, 42.0, 16.0), (58.0, 86.0, 18.0), (96.0, 94.0, 12.0), (26.0, 92.0, 11.0)];
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut v = 0.08 + 0.14 * (x as f32 / w as f32); // gentle uneven haze
            for &(cx, cy, r) in &centers {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                v += 0.85 * (-d2 / (2.0 * (r * 0.55f32).powi(2))).exp();
            }
            // subtle deterministic grain
            let hval = ((x * 73856093) ^ (y * 19349663)) & 0xff;
            v += (hval as f32 / 255.0 - 0.5) * 0.08;
            out[y * w + x] = v.clamp(0.0, 1.0);
        }
    }
    out
}

fn render_op_preview_bgra(patch: &[f32], w: usize, h: usize, op: &Op, rgb: [f32; 3]) -> Vec<u8> {
    match op {
        Op::GaussianBlur { sigma } => render_mono_bgra(&blur_separable(patch, w, h, (*sigma as f32).clamp(0.6, 8.0)), rgb),
        Op::Median { radius } => render_mono_bgra(&preview_median(patch, w, h, (*radius).clamp(1, 3) as isize), rgb),
        Op::UnsharpMask { sigma, amount } => {
            let b = blur_separable(patch, w, h, (*sigma as f32).clamp(0.6, 6.0));
            let a = (*amount as f32).clamp(0.2, 5.0);
            let out: Vec<f32> = patch.iter().zip(b.iter()).map(|(&x, &y)| (x + a * (x - y)).clamp(0.0, 1.0)).collect();
            render_mono_bgra(&out, rgb)
        }
        Op::DifferenceOfGaussians { low_sigma, high_sigma } => {
            let lo = blur_separable(patch, w, h, (*low_sigma as f32).clamp(0.5, 4.0));
            let hi = blur_separable(patch, w, h, (*high_sigma as f32).clamp(1.5, 8.0));
            let mut out: Vec<f32> = lo.iter().zip(hi.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect();
            normalize_peak(&mut out);
            render_mono_bgra(&out, rgb)
        }
        Op::Sobel {} | Op::MorphGradient { .. } => render_mono_bgra(&preview_sobel(patch, w, h), rgb),
        Op::Laplace {} | Op::Frangi { .. } | Op::Sato { .. } => render_mono_bgra(&preview_laplace(patch, w, h), rgb),
        Op::RollingBall { radius } => render_mono_bgra(&preview_rolling_ball(patch, w, h, ((*radius / 2.0).round() as isize).clamp(3, 14)), rgb),
        Op::Clahe { clip_limit, .. } => render_mono_bgra(&preview_clahe(patch, w, h, (*clip_limit as f32).clamp(0.5, 6.0)), rgb),
        Op::WhiteTophat { radius } => render_mono_bgra(&preview_tophat(patch, w, h, (*radius as isize).clamp(2, 8)), rgb),
        Op::BlackTophat { radius } => {
            let inv: Vec<f32> = patch.iter().map(|&v| 1.0 - v).collect();
            render_mono_bgra(&preview_tophat(&inv, w, h, (*radius as isize).clamp(2, 8)), rgb)
        }
        Op::Invert {} => render_mono_bgra(&patch.iter().map(|&v| 1.0 - v.clamp(0.0, 1.0)).collect::<Vec<_>>(), rgb),
        Op::LogContrast { gain } => {
            let g = (*gain as f32).clamp(0.5, 6.0);
            let denom = (1.0 + 9.0 * g).ln();
            let out: Vec<f32> = patch.iter().map(|&v| ((1.0 + v.clamp(0.0, 1.0) * 9.0 * g).ln() / denom).clamp(0.0, 1.0)).collect();
            render_mono_bgra(&out, rgb)
        }
        Op::SigmoidContrast { .. } => {
            let out: Vec<f32> = patch.iter().map(|&v| 1.0 / (1.0 + (-8.0 * (v - 0.5)).exp())).collect();
            render_mono_bgra(&out, rgb)
        }
        Op::OtsuThreshold { .. } | Op::Threshold { .. } | Op::SauvolaThreshold { .. } | Op::NiblackThreshold { .. } => {
            render_mask_overlay_bgra(patch, &preview_otsu(patch), rgb)
        }
        Op::LiThreshold { .. } => render_mask_overlay_bgra(patch, &preview_li(patch), rgb),
        Op::TriangleThreshold { .. } | Op::HysteresisThreshold { .. } => render_mask_overlay_bgra(patch, &preview_triangle(patch), rgb),
        Op::Canny { sigma } => render_mask_overlay_bgra(patch, &preview_canny(patch, w, h, (*sigma as f32).clamp(0.6, 4.0)), rgb),
        Op::FillHoles {} | Op::RemoveSmall { .. } | Op::RemoveSmallHoles { .. } | Op::Open { .. } | Op::Close { .. } | Op::Dilate { .. } | Op::Erode { .. } => {
            let m = morph_dilate(&morph_erode(&preview_otsu(patch), w, h, 1), w, h, 1);
            render_mask_overlay_bgra(patch, &m, rgb)
        }
        Op::Label {} | Op::Watershed { .. } | Op::ExpandLabels { .. } | Op::ClearBorder { .. } | Op::FindBoundaries {} | Op::Measure {} => {
            let m = preview_otsu(patch);
            let edges = preview_sobel(&m, w, h);
            let ring: Vec<f32> = edges.into_iter().map(|v| if v > 0.25 { 1.0 } else { 0.0 }).collect();
            render_mask_overlay_bgra(patch, &ring, rgb)
        }
        _ => render_mono_bgra(&blur_separable(patch, w, h, 1.5), rgb),
    }
}

fn normalize_peak(out: &mut [f32]) {
    let mx = out.iter().copied().fold(1e-6f32, f32::max);
    for v in out {
        *v = (*v / mx).clamp(0.0, 1.0);
    }
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

fn preview_median(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let mut buf = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for y in 0..h {
        for x in 0..w {
            buf.clear();
            for dy in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for dx in -r..=r {
                    let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    buf.push(src[sy * w + sx]);
                }
            }
            buf.sort_unstable_by(|a, b| a.total_cmp(b));
            out[y * w + x] = buf[buf.len() / 2];
        }
    }
    out
}

fn preview_sobel(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let at = |x: isize, y: isize| src[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let gx = -at(x - 1, y - 1) + at(x + 1, y - 1) - 2.0 * at(x - 1, y) + 2.0 * at(x + 1, y) - at(x - 1, y + 1) + at(x + 1, y + 1);
            let gy = -at(x - 1, y - 1) - 2.0 * at(x, y - 1) - at(x + 1, y - 1) + at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1);
            out[y as usize * w + x as usize] = gx.hypot(gy);
        }
    }
    normalize_peak(&mut out);
    out
}

fn preview_laplace(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let at = |x: isize, y: isize| src[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let v = (4.0 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)).abs();
            out[y as usize * w + x as usize] = v;
        }
    }
    normalize_peak(&mut out);
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

fn preview_rolling_ball(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let bg = morph_dilate(&morph_erode(src, w, h, r), w, h, r);
    let bg = blur_separable(&bg, w, h, 3.0);
    let mut out: Vec<f32> = src.iter().zip(bg.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect();
    normalize_peak(&mut out);
    out
}

fn preview_tophat(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let opened = morph_dilate(&morph_erode(src, w, h, r), w, h, r);
    let mut out: Vec<f32> = src.iter().zip(opened.iter()).map(|(&a, &b)| (a - b).max(0.0)).collect();
    normalize_peak(&mut out);
    out
}

fn preview_clahe(src: &[f32], w: usize, h: usize, clip: f32) -> Vec<f32> {
    let mean = blur_separable(src, w, h, 8.0);
    let sq: Vec<f32> = src.iter().zip(mean.iter()).map(|(&a, &m)| (a - m) * (a - m)).collect();
    let var = blur_separable(&sq, w, h, 8.0);
    let gain = (clip * 0.45).clamp(0.4, 2.5);
    src.iter()
        .zip(mean.iter().zip(var.iter()))
        .map(|(&a, (&m, &v))| {
            let std = v.sqrt().max(0.06);
            (0.5 + gain * (a - m) / (2.2 * std)).clamp(0.0, 1.0)
        })
        .collect()
}

fn histogram_256(src: &[f32]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &v in src {
        let bin = (v.clamp(0.0, 1.0) * 255.0).round() as usize;
        h[bin.min(255)] += 1;
    }
    h
}

fn preview_otsu(src: &[f32]) -> Vec<f32> {
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

fn preview_li(src: &[f32]) -> Vec<f32> {
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

fn preview_triangle(src: &[f32]) -> Vec<f32> {
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

fn preview_canny(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let smooth = blur_separable(src, w, h, sigma);
    let grad = preview_sobel(&smooth, w, h);
    grad.into_iter().map(|v| if v > 0.22 { 1.0 } else { 0.0 }).collect()
}
