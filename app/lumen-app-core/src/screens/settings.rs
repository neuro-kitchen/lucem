//! Settings (Ctrl+,): sections on the left, their settings on the right. Every setting says what
//! it does and what its default is, and takes effect at once (the theme too).

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Icon, Selectable as _, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Window};

use crate::data::prefs::Prefs;
use crate::data::store::Store;
use crate::ui::theme::{self, color, ACCENTS, PASTEBOARDS};
use crate::ui::widgets::stepper;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Appearance,
    Views,
    Compute,
    Models,
    Performance,
    About,
}

impl Section {
    const ALL: [Section; 6] = [Section::Appearance, Section::Views, Section::Compute, Section::Models, Section::Performance, Section::About];

    fn title(self) -> &'static str {
        match self {
            Section::Appearance => "Appearance",
            Section::Views => "Views",
            Section::Compute => "Compute",
            Section::Models => "Models",
            Section::Performance => "Performance",
            Section::About => "About",
        }
    }

    fn icon(self) -> IconName {
        match self {
            Section::Appearance => IconName::Palette,
            Section::Views => IconName::LayoutGrid,
            Section::Compute => IconName::Cpu,
            Section::Models => IconName::Sparkles,
            Section::Performance => IconName::Gauge,
            Section::About => IconName::Info,
        }
    }
}

pub struct SettingsView {
    store: Entity<Store>,
    section: Section,
    on_close: crate::screens::palette::Run,
}

impl SettingsView {
    pub fn new(store: Entity<Store>, on_close: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        Self { store, section: Section::Appearance, on_close: Rc::new(on_close) }
    }

    /// Changes the preferences; the look follows at once.
    fn set(store: &Entity<Store>, f: impl FnOnce(&mut Prefs), window: &mut Window, cx: &mut App) {
        store.update(cx, |s, cx| s.update_prefs(f, cx));
        let prefs = store.read(cx).prefs.clone();
        theme::configure(&prefs, window);
        theme::apply(window, cx);
    }

    /// A setting: its name, what it does and its default on the left, the control on the right.
    fn row(title: &str, what: &str, default: &str, control: impl IntoElement) -> impl IntoElement {
        h_flex()
            .w_full()
            .py(px(theme::MD))
            .gap(px(theme::LG))
            .border_b_1()
            .border_color(color::border())
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap(px(2.))
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(title.to_string()))
                    .child(div().text_xs().text_color(color::text_muted()).child(what.to_string()))
                    .child(div().text_xs().text_color(color::text_disabled()).child(format!("Default: {default}"))),
            )
            .child(div().flex_none().child(control))
    }

    /// Mutually exclusive choices as one row of buttons.
    fn choices(id: &'static str, options: &[&str], current: usize, on_pick: impl Fn(usize, &mut Window, &mut App) + 'static) -> impl IntoElement {
        let on_pick = Rc::new(on_pick);
        h_flex().gap(px(2.)).p(px(2.)).rounded(px(6.)).bg(color::control()).children(options.iter().enumerate().map(|(i, o)| {
            let f = on_pick.clone();
            Button::new(SharedString::from(format!("{id}-{i}"))).ghost().small().label(o.to_string()).selected(i == current).on_click(move |_, w, cx| f(i, w, cx))
        }))
    }

    fn appearance(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let theme_now = match p.theme {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        };
        let s1 = st.clone();
        let theme = Self::choices("set-theme", &["System", "Dark", "Light"], theme_now, move |i, w, cx| Self::set(&s1, |p| p.theme = [None, Some(true), Some(false)][i], w, cx));
        let accents = h_flex().gap(px(theme::XS)).children(ACCENTS.iter().map(|(name, rgb)| {
            let (s2, rgb) = (st.clone(), *rgb);
            let on = p.accent == rgb;
            div()
                .id(SharedString::from(format!("set-accent-{name}")))
                .size(px(22.))
                .rounded_full()
                .cursor_pointer()
                .bg(gpui_kit::rgb(rgb))
                .border_2()
                .border_color(if on { color::text() } else { color::border() })
                .on_click(move |_, w, cx| Self::set(&s2, |p| p.accent = rgb, w, cx))
        }));
        let s3 = st.clone();
        let paste = Self::choices("set-paste", &PASTEBOARDS, p.pasteboard as usize, move |i, w, cx| Self::set(&s3, |p| p.pasteboard = i as u8, w, cx));
        let s4 = st.clone();
        let size = stepper("set-font", format!("{:.0} px", p.font_size), move |d, w, cx| Self::set(&s4, |p| p.font_size = (p.font_size + d).clamp(12.0, 18.0), w, cx));
        v_flex()
            .child(Self::row("Theme", "Dark or light, or as the system is set (follows it while running).", "Dark", theme))
            .child(Self::row("Accent", "The colour of selections, the active tool and primary buttons.", "Blue", accents))
            .child(Self::row("Pasteboard", "What is around the image. Grey shows the edges of dark images.", "Grey", paste))
            .child(Self::row("Interface size", "Text size of the whole interface (drag the number).", "15 px", size))
            .into_any_element()
    }

    fn views(&self, p: &Prefs, cx: &App) -> AnyElement {
        let st = self.store.clone();
        let tool = self.store.read(cx).tool.clone();
        let toggle = |id: &'static str, on: bool, f: crate::screens::palette::Run| Self::choices(id, &["Off", "On"], usize::from(on), move |i, w, cx| {
            if (i == 1) != on {
                f(w, cx)
            }
        });
        let (s1, s2, s3, s4) = (st.clone(), st.clone(), st.clone(), st.clone());
        v_flex()
            .child(Self::row("Rulers", "Rulers along the edges of a plane (Ctrl+R).", "Off", toggle("set-rulers", tool.rulers, Rc::new(move |_, cx| s1.update(cx, |s, cx| s.update_tool(|t| t.rulers = !t.rulers, cx))))))
            .child(Self::row("Crosshair", "The 3D cursor drawn across planes (the Probe tool moves it).", "Off", toggle("set-cross", tool.crosshair, Rc::new(move |_, cx| s2.update(cx, |s, cx| s.update_tool(|t| t.crosshair = !t.crosshair, cx))))))
            .child(Self::row("Link views", "Views of the same image move and zoom together.", "Off", toggle("set-link", p.link_views, Rc::new(move |_, cx| s3.update(cx, |s, cx| s.update_prefs(|p| p.link_views = !p.link_views, cx))))))
            .child(Self::row("Results over the image", "The pipeline's results drawn on the planes.", "On", toggle("set-results", tool.show_results, Rc::new(move |_, cx| s4.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx))))))
            .into_any_element()
    }

    /// The compute backend: chosen here, switched while running, tested with a checked read.
    fn compute(&self, p: &Prefs, cx: &App) -> AnyElement {
        let s = self.store.read(cx);
        let targets: Vec<lumen_core::ComputeTarget> = lumen_core::ComputeTarget::available();
        let mut names: Vec<String> = vec!["Automatic".into()];
        names.extend(targets.iter().map(|t| match t.name() {
            "wgpu" => "wgpu (Vulkan, Metal, DX12)".to_string(),
            "cuda" => "CUDA".to_string(),
            "cpu" => "CPU".to_string(),
            "hip" => "HIP (ROCm)".to_string(),
            other => other.to_string(),
        }));
        let current = p.runtime.as_deref().and_then(|r| targets.iter().position(|t| t.name() == r)).map_or(0, |i| i + 1);
        let options: Vec<&str> = names.iter().map(String::as_str).collect();
        let (st, keys) = (self.store.clone(), targets.iter().map(|t| t.name().to_string()).collect::<Vec<_>>());
        let pick = Self::choices_owned("set-runtime", &options, current, move |i, _, cx| {
            let r = if i == 0 { None } else { keys.get(i - 1).cloned() };
            st.update(cx, |s, cx| s.set_runtime(r, cx));
        });
        let device = match (&s.device, s.prefs.runtime.as_deref()) {
            (Some(d), _) => d.label.clone(),
            (None, _) => "Not started (it starts when a result or the 3D view needs it)".to_string(),
        };
        let st = self.store.clone();
        let test = Button::new("set-test-device").outline().small().icon(Icon::new(IconName::Play)).label("Test").on_click(move |_, _, cx| st.update(cx, |s, cx| s.test_device(cx)));
        v_flex()
            .child(Self::row("Backend", "Where images are computed. Switching takes effect at once: results are computed again on the new device.", "Automatic (the first that starts: GPU first)", pick))
            .child(Self::row("Device", &device, "—", div()))
            .child(Self::row("Test", "A blur on the device, read back and checked (a wrong or all-zero result is reported, never trusted).", "—", test))
            .child(div().pt(px(theme::SM)).text_xs().text_color(color::text_muted()).child(format!("Compiled in: {}. The command line and plugins also read LUMEN_RUNTIME.", targets.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "))))
            .into_any_element()
    }

    /// The model plugins found, their models, installing them and choosing the one used.
    fn models(&self, p: &Prefs, cx: &App) -> AnyElement {
        let s = self.store.read(cx);
        let st = self.store.clone();
        let rescan = Button::new("set-ml-rescan").ghost().small().icon(Icon::new(IconName::RefreshCw)).label("Look again").on_click(move |_, _, cx| st.update(cx, |s, cx| s.scan_plugins(cx)));
        let chosen = s.ml.segmenter(p.ml_model.as_deref()).map(|(_, id, _)| id);
        let busy = s.ml.busy.clone();
        let mut col = v_flex().gap(px(theme::MD)).child(
            h_flex()
                .gap(px(theme::SM))
                .child(div().flex_1().text_xs().text_color(color::text_muted()).child("Models come as plugins: separate programs (lumen-ml-*), each with its own licence and runtime, so they never weigh on lumen."))
                .child(rescan),
        );
        if let Some((what, f)) = &busy {
            col = col.child(div().text_sm().text_color(color::accent_hover()).child(format!("{what}{}", f.map_or(String::new(), |f| format!(" — {:.0} %", f * 100.0)))));
        }
        if s.ml.scanning || !s.ml.scanned {
            col = col.child(div().text_sm().text_color(color::text_muted()).child("Looking for plugins…"));
        } else if s.ml.plugins.is_empty() {
            col = col.child(div().text_sm().child("No model plugin found.")).child(div().text_xs().text_color(color::text_muted()).child("Build one (cargo build -p lumen-ml-sam: it lands next to lumen-app) or copy a lumen-ml-* program into one of:"));
            for d in lumen_ml::plugin_dirs().into_iter().take(3) {
                col = col.child(div().text_xs().font_family("monospace").text_color(color::text_muted()).child(d.display().to_string()));
            }
        }
        for (pi, plugin) in s.ml.plugins.iter().enumerate() {
            let mut card = v_flex()
                .gap(px(theme::SM))
                .p(px(theme::MD))
                .rounded(px(8.))
                .border_1()
                .border_color(color::border())
                .bg(color::raised())
                .child(h_flex().gap(px(theme::SM)).child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(plugin.name.clone())).child(div().text_xs().text_color(color::text_muted()).child(format!("{} · {}", plugin.version, plugin.licence))))
                .child(div().text_xs().text_color(if plugin.runtime.ready { color::text_muted() } else { color::warning() }).child(plugin.runtime.detail.clone()));
            if !plugin.runtime.ready {
                let (st, program) = (self.store.clone(), plugin.program.clone());
                card = card.child(h_flex().child(
                    Button::new(SharedString::from(format!("set-ml-runtime-{pi}"))).small().primary().icon(Icon::new(IconName::Download)).label("Install runtime").disabled(busy.is_some()).on_click(move |_, _, cx| st.update(cx, |s, cx| s.install_model(program.clone(), lumen_ml::RUNTIME_ID.to_string(), cx))),
                ));
            }
            for (mi, m) in plugin.models.iter().enumerate() {
                let size = if m.download_bytes() > 0 { format!(" · {:.0} MB", m.download_bytes() as f64 / 1e6) } else { String::new() };
                let (st, program, id) = (self.store.clone(), plugin.program.clone(), m.id.clone());
                let action = if m.installed {
                    let on = chosen.as_deref() == Some(m.id.as_str());
                    Button::new(SharedString::from(format!("set-ml-use-{pi}-{mi}"))).small().ghost().label(if on { "In use" } else { "Use" }).selected(on).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_ml_model(id.clone(), cx)))
                } else {
                    Button::new(SharedString::from(format!("set-ml-install-{pi}-{mi}"))).small().outline().icon(Icon::new(IconName::Download)).label("Install").disabled(busy.is_some() || !plugin.runtime.ready).on_click(move |_, _, cx| st.update(cx, |s, cx| s.install_model(program.clone(), id.clone(), cx)))
                };
                card = card.child(
                    h_flex()
                        .gap(px(theme::MD))
                        .py(px(2.))
                        .child(v_flex().flex_1().min_w_0().child(div().text_sm().child(m.name.clone())).child(div().text_xs().text_color(color::text_muted()).truncate().child(format!("{} · {}{size}", m.task.title(), m.description))))
                        .child(action),
                );
            }
            col = col.child(card);
        }
        for p in &s.ml.problems {
            col = col.child(div().text_xs().text_color(color::warning()).child(p.clone()));
        }
        col.into_any_element()
    }

    /// [`Self::choices`] for labels made at run time.
    fn choices_owned(id: &'static str, options: &[&str], current: usize, on_pick: impl Fn(usize, &mut Window, &mut App) + 'static) -> impl IntoElement {
        let on_pick = Rc::new(on_pick);
        h_flex().flex_wrap().gap(px(2.)).p(px(2.)).rounded(px(6.)).bg(color::control()).children(options.iter().enumerate().map(|(i, o)| {
            let f = on_pick.clone();
            Button::new(SharedString::from(format!("{id}-{i}"))).ghost().small().label(o.to_string()).selected(i == current).on_click(move |_, w, cx| f(i, w, cx))
        }))
    }

    fn performance(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let auto = lumen_view::TileService::default_threads();
        let threads = if p.tile_threads == 0 { format!("Auto ({auto})") } else { p.tile_threads.to_string() };
        let threads = stepper("set-threads", threads, move |d, _, cx| st.update(cx, |s, cx| s.update_prefs(|p| p.tile_threads = (p.tile_threads as i64 + d as i64).clamp(0, 32) as usize, cx)));
        v_flex()
            .child(Self::row("Tile readers", "Threads reading image tiles for the first view (new views use 2). Applies to views opened after the change; 0 is automatic.", &format!("Auto ({auto})"), threads))
            .into_any_element()
    }

    fn about(&self, p: &Prefs) -> AnyElement {
        let st = self.store.clone();
        let path = Prefs::path().map_or("—".to_string(), |p| p.display().to_string());
        let reset = Button::new("set-reset").outline().small().label("Reset all settings").on_click(move |_, w, cx| {
            Self::set(
                &st,
                |p| {
                    let recent = std::mem::take(&mut p.recent);
                    *p = Prefs { recent, ..Prefs::default() };
                },
                w,
                cx,
            )
        });
        let _ = p;
        v_flex()
            .child(Self::row("Lumen", "GPU workstation for histology and radiology images.", env!("CARGO_PKG_VERSION"), div()))
            .child(Self::row("Settings file", &path, "per user", div()))
            .child(Self::row("Reset", "Every setting back to its default (recent files stay).", "—", reset))
            .into_any_element()
    }
}

impl Render for SettingsView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let prefs = self.store.read(cx).prefs.clone();
        let nav = v_flex().flex_none().w(px(190.)).h_full().p(px(theme::SM)).gap(px(2.)).bg(color::chrome()).border_r_1().border_color(color::border()).children(Section::ALL.into_iter().map(|sec| {
            Button::new(SharedString::from(format!("settings-{}", sec.title()))).ghost().small().w_full().icon(Icon::new(sec.icon())).label(sec.title()).selected(sec == self.section).on_click(cx.listener(move |this, _, _, cx| {
                this.section = sec;
                cx.notify();
            }))
        }));
        let body = match self.section {
            Section::Appearance => self.appearance(&prefs),
            Section::Views => self.views(&prefs, cx),
            Section::Compute => self.compute(&prefs, cx),
            Section::Models => self.models(&prefs, cx),
            Section::Performance => self.performance(&prefs),
            Section::About => self.about(&prefs),
        };
        let (close_a, close_b) = (self.on_close.clone(), self.on_close.clone());
        div()
            .id("settings-backdrop")
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::black().opacity(0.45))
            .on_click(move |_, w, cx| close_a(w, cx))
            .child(
                v_flex()
                    .id("settings")
                    .w(px(780.))
                    .h(px(540.))
                    .rounded(px(10.))
                    .overflow_hidden()
                    .border_1()
                    .border_color(color::border())
                    .bg(color::panel())
                    .shadow_lg()
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .flex_none()
                            .h(px(44.))
                            .px(px(theme::LG))
                            .border_b_1()
                            .border_color(color::border())
                            .child(div().text_base().font_weight(FontWeight::SEMIBOLD).child("Settings"))
                            .child(div().flex_1())
                            .child(div().text_xs().text_color(color::text_muted()).child("Changes apply at once · Esc closes"))
                            .child(Button::new("settings-close").ghost().small().icon(Icon::new(IconName::X)).on_click(move |_, w, cx| close_b(w, cx))),
                    )
                    .child(h_flex().flex_1().min_h_0().child(nav).child(v_flex().id("settings-body").flex_1().min_w_0().h_full().overflow_y_scroll().px(px(theme::XL)).py(px(theme::SM)).child(div().text_lg().font_weight(FontWeight::SEMIBOLD).py(px(theme::SM)).child(self.section.title())).child(body))),
            )
    }
}
