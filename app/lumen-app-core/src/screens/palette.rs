//! Search (Ctrl+K, F3; Spotlight / Blender's F3): every action by name — workspaces, files,
//! tools, every operator — filtered as you type, shown as the "Add view or container" dialog
//! shows its rows: an icon tile, the name, a line saying what it does, under group headings.
//! ↑ ↓ move, Enter runs the highlighted one, a click runs any, Escape closes. Opened from
//! Process ▸ "Find an operation…", it lists the operations only.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Icon};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, EventEmitter, FontWeight, InteractiveElement as _, IntoElement, KeyDownEvent, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window,
};

use crate::ui::theme::{self, color};

pub type Run = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Clone)]
pub struct Command {
    pub group: &'static str,
    pub name: String,
    /// What it does, or its key / signature (the line under the name).
    pub hint: String,
    pub icon: IconName,
    pub run: Run,
}

pub enum PaletteEvent {
    Close,
    /// Close, then run this (the app runs it after the search is gone, outside any update, so
    /// what it opens — Settings, a dialog — keeps the focus).
    Run(Run),
}

pub struct Palette {
    input: Entity<InputState>,
    commands: Vec<Command>,
    title: &'static str,
    /// The highlighted row (of the filtered list).
    cursor: usize,
    _subs: Vec<Subscription>,
}

impl EventEmitter<PaletteEvent> for Palette {}

/// Rows listed at most.
const SHOWN: usize = 40;

impl Palette {
    pub fn new(commands: Vec<Command>, title: &'static str, placeholder: &'static str, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder(placeholder));
        let subs = vec![cx.subscribe_in(&input, window, |this, _, e: &InputEvent, window, cx| match e {
            InputEvent::PressEnter { .. } => this.run_cursor(window, cx),
            InputEvent::Change => {
                this.cursor = 0;
                cx.notify()
            }
            _ => {}
        })];
        let focus = gpui_kit::Focusable::focus_handle(input.read(cx), cx);
        window.focus(&focus, cx);
        Self { input, commands, title, cursor: 0, _subs: subs }
    }

    /// Commands whose group, name and hint hold every word typed.
    fn matches(&self, cx: &App) -> Vec<Command> {
        let q = self.input.read(cx).value().to_lowercase();
        let words: Vec<&str> = q.split_whitespace().collect();
        self.commands
            .iter()
            .filter(|c| {
                let hay = format!("{} {} {}", c.group, c.name, c.hint).to_lowercase();
                words.iter().all(|w| hay.contains(w))
            })
            .take(SHOWN)
            .cloned()
            .collect()
    }

    fn run_cursor(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let _ = window;
        if let Some(c) = self.matches(cx).into_iter().nth(self.cursor) {
            cx.emit(PaletteEvent::Run(c.run.clone()));
        }
    }

    fn on_key(&mut self, e: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let n = self.matches(cx).len();
        match e.keystroke.key.as_str() {
            "down" => self.cursor = (self.cursor + 1).min(n.saturating_sub(1)),
            "up" => self.cursor = self.cursor.saturating_sub(1),
            "escape" => cx.emit(PaletteEvent::Close),
            "enter" => {
                self.run_cursor(window, cx);
                cx.stop_propagation();
                return;
            }
            _ => return,
        }
        cx.stop_propagation();
        cx.notify();
    }
}

impl Render for Palette {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let found = self.matches(cx);
        let cursor = self.cursor.min(found.len().saturating_sub(1));
        let mut list = v_flex().id("palette-rows").flex_1().min_h_0().overflow_y_scroll();
        let mut last_group = "";
        for (i, c) in found.into_iter().enumerate() {
            if c.group != last_group {
                last_group = c.group;
                list = list.child(div().px(px(theme::SM)).pt(px(if i == 0 { theme::XS } else { theme::MD })).pb(px(theme::XS)).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(color::text_muted()).child(c.group));
            }
            let run = c.run.clone();
            let me = cx.entity().downgrade();
            list = list.child(
                h_flex()
                    .id(("palette-row", i))
                    .min_h(px(48.))
                    .px(px(theme::SM))
                    .gap(px(theme::MD))
                    .rounded(px(6.))
                    .cursor_pointer()
                    .when(i == cursor, |d| d.bg(color::control()))
                    .hover(|d| d.bg(color::control()))
                    .child(div().flex_none().size(px(32.)).rounded(px(6.)).border_1().border_color(color::hover()).flex().items_center().justify_center().child(Icon::new(c.icon)))
                    .child(v_flex().flex_1().min_w_0().child(div().text_sm().font_weight(FontWeight::SEMIBOLD).truncate().child(SharedString::from(c.name.clone()))).child(div().text_xs().text_color(color::text_muted()).truncate().child(c.hint.clone())))
                    .on_click(move |_, _, cx| {
                        let run = run.clone();
                        let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Run(run)));
                    }),
            );
        }
        let me = cx.entity().downgrade();
        let close = cx.entity().downgrade();
        div()
            .id("palette-backdrop")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .items_start()
            .justify_center()
            .pt(px(72.))
            .px(px(theme::LG))
            .bg(gpui_kit::black().opacity(0.35))
            .on_click(move |_, _, cx| {
                let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
            })
            .child(
                v_flex()
                    .id("palette")
                    .occlude()
                    // Narrows with the window, never wider than reads well
                    .w_full()
                    .max_w(px(520.))
                    .max_h(gpui_kit::relative(0.8))
                    .p(px(theme::SM))
                    .gap(px(theme::XS))
                    .rounded(px(10.))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .shadow_lg()
                    .capture_key_down(cx.listener(Self::on_key))
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(
                        h_flex()
                            .px(px(theme::SM))
                            .py(px(theme::XS))
                            .gap(px(theme::SM))
                            .items_baseline()
                            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(self.title))
                            .child(div().text_xs().text_color(color::text_muted()).child("↑ ↓ Enter · Esc"))
                            .child(div().flex_1())
                            .child(
                                div()
                                    .id("palette-close")
                                    .cursor_pointer()
                                    .rounded(px(4.))
                                    .p(px(2.))
                                    .hover(|d| d.bg(color::control()))
                                    .child(Icon::new(IconName::X))
                                    .on_click(move |_, _, cx| {
                                        let _ = close.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
                                    }),
                            ),
                    )
                    .child(Input::new(&self.input))
                    .child(list),
            )
    }
}
