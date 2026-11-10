//! The home view (no image yet): open an image, an OME-Zarr container or a project; recent
//! files; samples (generated, and what is in the workspace's `.data/`).

use std::path::PathBuf;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, App, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _, Window};
use lumen_io::{discover_workspace_data_paths, PhantomKind};

use crate::data::store::{display_name, sample_name, Store};
use crate::ui::theme::{self, color};

/// The home view. `open_file` and `open_project` open the dialogs.
pub fn home(store: &Entity<Store>, open_file: impl Fn(&mut Window, &mut App) + 'static, open_project: impl Fn(&mut Window, &mut App) + 'static, cx: &App) -> impl IntoElement {
    let s = store.read(cx);
    let recent = s.prefs.recent.clone();
    let opening = s.opening.clone();
    let data: Vec<PathBuf> = std::env::current_dir().map(|d| discover_workspace_data_paths(&d)).unwrap_or_default();
    let entry = |id: SharedString, icon: IconName, label: String, sub: String, path: PathBuf| {
        let store = store.clone();
        h_flex()
            .id(id)
            .w_full()
            .h(px(36.))
            .px(px(theme::SM))
            .gap(px(theme::SM))
            .rounded(px(5.))
            .cursor_pointer()
            .hover(|d| d.bg(color::control()))
            .child(Icon::new(icon).small().text_color(color::text_muted()))
            .child(div().flex_none().max_w(px(260.)).truncate().text_sm().child(label))
            .child(div().flex_1().min_w_0().truncate().text_xs().text_color(color::text_muted()).child(sub))
            .on_click(move |_, _, cx| store.update(cx, |s, cx| s.open(path.clone(), cx)))
    };
    let column = |title: &'static str| v_flex().w(px(440.)).gap(px(2.)).child(div().pb(px(theme::XS)).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(title.to_uppercase()));

    let mut recent_col = column("Recent");
    if recent.is_empty() {
        recent_col = recent_col.child(div().px(px(theme::SM)).text_sm().text_color(color::text_muted()).child("Nothing yet."));
    }
    for (i, p) in recent.iter().take(8).enumerate() {
        let is_project = p.to_string_lossy().ends_with(".lumen.json");
        let label = if is_project { p.parent().map_or_else(|| display_name(p), display_name) } else { display_name(p) };
        let sub = p.parent().map(|d| d.display().to_string()).unwrap_or_default();
        recent_col = recent_col.child(entry(SharedString::from(format!("recent-{i}")), if is_project { IconName::FolderOpen } else { IconName::Image }, label, sub, p.clone()));
    }
    let mut samples = column("Samples");
    for (i, p) in data.iter().enumerate() {
        samples = samples.child(entry(SharedString::from(format!("data-{i}")), IconName::Microscope, display_name(p), "workspace .data/".into(), p.clone()));
    }
    samples = samples.child(entry(SharedString::from("sample-volume"), IconName::Box, "Sample volume".to_string(), "generated 256 × 256 × 128, two channels".into(), PathBuf::from("sample:volume")));
    for k in PhantomKind::ALL {
        let path = PathBuf::from(format!("sample:{}", sample_name(*k)));
        samples = samples.child(entry(SharedString::from(format!("sample-{}", sample_name(*k))), IconName::Sparkles, k.title().to_string(), "generated 2048 × 2048".into(), path));
    }

    v_flex()
        .id("home")
        .size_full()
        .items_center()
        .justify_center()
        .gap(px(theme::XL))
        .bg(color::canvas())
        .child(
            v_flex()
                .items_center()
                .gap(px(theme::SM))
                .child(div().text_size(px(28.)).font_weight(FontWeight::SEMIBOLD).child("Lumen"))
                .child(div().text_sm().text_color(color::text_muted()).child("GPU workstation for histology and radiology: open an image, annotate, build pipelines.")),
        )
        .child(
            h_flex()
                .gap(px(theme::SM))
                .child(Button::new("home-open").primary().icon(IconName::FolderOpen).label("Open image or OME-Zarr…").on_click(move |_, w, cx| open_file(w, cx)))
                .child(Button::new("home-project").outline().icon(IconName::Folder).label("Open project…").on_click(move |_, w, cx| open_project(w, cx))),
        )
        .when(!opening.is_empty(), |d| d.child(div().text_sm().text_color(color::warning()).child(format!("Opening {}…", opening.join(", ")))))
        .child(h_flex().items_start().gap(px(theme::XL)).child(recent_col).child(samples))
        .child(div().text_xs().text_color(color::text_muted()).child("Drop a file or folder anywhere · Ctrl+K searches every action · F1 lists the keys"))
        .child(Icon::new(IconName::Microscope).xsmall().text_color(color::canvas()))
}
