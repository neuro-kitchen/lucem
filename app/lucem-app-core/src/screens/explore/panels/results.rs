//! The Results tab: each result view of the shown image (visibility, opacity, colour map, state
//! of its whole-image analysis and of its layer in view).

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::{div, px, AnyElement, App, Entity, InteractiveElement as _, IntoElement, ParentElement as _, SharedString, StatefulInteractiveElement as _, Styled as _, StyledImage as _};

use crate::data::results::COLORMAPS;
use crate::data::store::Store;
use crate::domain::command::Command;
use crate::domain::graph::{NodeKind, ViewStyle};
use crate::navigation::route::Workspace;
use crate::ui::theme::{self, color};
use crate::ui::widgets::{row, stepper, MenuSelect, Section};

pub fn panel(store: &Entity<Store>, cx: &App) -> AnyElement {
    let s = store.read(cx);
    let Some(img) = s.doc.active_image.clone() else { return div().into_any_element() };
    let views: Vec<_> = s.doc.graph.views().filter(|v| s.doc.graph.images_of(&v.id).contains(&img)).cloned().collect();
    let st = store.clone();
    let open = Button::new("results-pipeline").ghost().xsmall().icon(Icon::new(IconName::Workflow)).tooltip("Open the Pipeline").on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_workspace(Workspace::Pipeline, cx)));
    let st = store.clone();
    let toggle = Button::new("results-toggle").ghost().xsmall().label(if s.tool.show_results { "Hide all" } else { "Show all" }).on_click(move |_, _, cx| st.update(cx, |s, cx| s.update_tool(|t| t.show_results = !t.show_results, cx)));
    let mut sec = Section::new("Results").extra(h_flex().child(toggle).child(open));
    if views.is_empty() {
        sec = sec.child(div().text_xs().text_color(color::text_muted()).child("No results yet (Process menu or Ctrl+K)."));
    }
    let analysis = s.analysis.clone();
    for v in views {
        let NodeKind::View { name, style } = &v.kind else { continue };
        let layer = s.results.get(&v.id);
        let err = analysis.as_ref().and_then(|a| a.errors.get(&v.id).cloned()).or_else(|| layer.and_then(|r| r.error.clone()));
        let table = analysis.as_ref().and_then(|a| a.tables.get(&v.id)).map(|t| t.rows.len());
        let status = match (&err, table, layer) {
            (Some(_), _, _) => "error".to_string(),
            (None, Some(n), _) => format!("{n} objects"),
            (None, None, Some(r)) => format!("{:.0} ms", r.ms),
            _ if s.computing => "…".into(),
            _ => String::new(),
        };
        let preview = analysis.as_ref().and_then(|a| a.previews.get(&v.id).cloned());
        let set = |f: fn(&mut ViewStyle, f32), x: f32| {
            let (st, id, name, style) = (store.clone(), v.id.clone(), name.clone(), style.clone());
            move |cx: &mut App| {
                let mut style = style.clone();
                f(&mut style, x);
                st.update(cx, |s, cx| {
                    let _ = s.execute(Command::SetView { id: id.clone(), name: name.clone(), style }, cx);
                });
            }
        };
        let eye = set(|st, _| st.visible = !st.visible, 0.0);
        let (more, less) = (set(|st, _| st.opacity = (st.opacity + 0.1).min(1.0), 0.0), set(|st, _| st.opacity = (st.opacity - 0.1).max(0.0), 0.0));
        let (st3, id3, n3, sty3) = (store.clone(), v.id.clone(), name.clone(), style.clone());
        let lut_idx = COLORMAPS.iter().position(|m| *m == style.colormap);
        sec = sec.child(
            v_flex()
                .w_full()
                .gap(px(theme::XS))
                .pb(px(theme::SM))
                .border_b_1()
                .border_color(color::border())
                .child(
                    h_flex()
                        .gap(px(theme::SM))
                        .child(Button::new(SharedString::from(format!("view-eye-{}", v.id))).ghost().xsmall().icon(Icon::new(if style.visible { IconName::Eye } else { IconName::EyeOff })).on_click(move |_, _, cx| eye(cx)))
                        .child(div().flex_1().min_w_0().truncate().text_sm().child(name.clone()))
                        .child(div().text_xs().text_color(if err.is_some() { color::danger() } else { color::text_muted() }).child(status)),
                )
                .children(preview.map(|p| div().w_full().h(px(96.)).rounded(px(4.)).overflow_hidden().bg(color::canvas()).child(gpui_kit::img(p).size_full().object_fit(gpui_kit::ObjectFit::Contain))))
                .child(row("Opacity", stepper(&format!("view-opacity-{}", v.id), format!("{:.0} %", style.opacity * 100.0), move |d, _, cx| if d > 0.0 { more(cx) } else { less(cx) })))
                .child(row(
                    "Colours",
                    MenuSelect::new(SharedString::from(format!("view-lut-{}", v.id)), COLORMAPS.iter().map(|m| SharedString::from(*m)).collect(), lut_idx, move |i, _, cx| {
                        let style = ViewStyle { colormap: COLORMAPS[i].to_string(), ..sty3.clone() };
                        st3.update(cx, |s, cx| {
                            let _ = s.execute(Command::SetView { id: id3.clone(), name: n3.clone(), style }, cx);
                        });
                    }),
                ))
                .children(err.map(|e| div().text_xs().text_color(color::danger()).child(e))),
        );
    }
    let footer = analysis.map(|a| div().text_xs().text_color(color::text_muted()).child(format!("Whole image analysed at level {} ({:.0} ms): Otsu thresholds and tables come from it.", a.level, a.ms)));
    v_flex().id("results-panel").size_full().overflow_y_scroll().p(px(theme::MD)).gap(px(theme::LG)).child(sec).children(footer).into_any_element()
}
