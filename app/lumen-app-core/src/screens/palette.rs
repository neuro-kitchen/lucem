//! Search (Ctrl+K, F3; Blender's F3 search, from dsp-kitchen's command palette): every action by
//! name — workspaces, files, tools, every operator — filtered as you type; Enter runs the first,
//! a click runs any, Escape closes.

use std::rc::Rc;

use gpui_kit::component::h_flex;
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::v_flex;
use gpui_kit::{
    div, px, App, AppContext as _, Context, Entity, EventEmitter, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _,
    Subscription, Window,
};

use crate::ui::theme::{self, color};

pub type Run = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Clone)]
pub struct Command {
    pub group: &'static str,
    pub name: String,
    /// A short note at the right (a key, a signature).
    pub hint: String,
    pub run: Run,
}

pub enum PaletteEvent {
    Close,
}

pub struct Palette {
    input: Entity<InputState>,
    commands: Vec<Command>,
    _subs: Vec<Subscription>,
}

impl EventEmitter<PaletteEvent> for Palette {}

const SHOWN: usize = 14;

impl Palette {
    pub fn new(commands: Vec<Command>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("Search an operation, a tool or an action…"));
        let subs = vec![cx.subscribe_in(&input, window, |this, _, e: &InputEvent, window, cx| match e {
            InputEvent::PressEnter { .. } => {
                if let Some(c) = this.matches(cx).into_iter().next() {
                    cx.emit(PaletteEvent::Close);
                    (c.run)(window, cx);
                }
            }
            InputEvent::Change => cx.notify(),
            _ => {}
        })];
        let focus = gpui_kit::Focusable::focus_handle(input.read(cx), cx);
        window.focus(&focus, cx);
        Self { input, commands, _subs: subs }
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
}

impl Render for Palette {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.matches(cx).into_iter().enumerate().map(|(i, c)| {
            let run = c.run.clone();
            let me = cx.entity().downgrade();
            h_flex()
                .id(("palette-row", i))
                .h(px(30.))
                .px(px(theme::MD))
                .gap(px(theme::MD))
                .rounded(px(5.))
                .cursor_pointer()
                .hover(|d| d.bg(color::control()))
                .when_first(i == 0)
                .child(div().w(px(90.)).flex_none().text_xs().text_color(color::text_muted()).child(c.group))
                .child(div().flex_1().min_w_0().truncate().text_sm().child(SharedString::from(c.name.clone())))
                .child(div().flex_none().text_xs().font_family("monospace").text_color(color::text_muted()).child(c.hint.clone()))
                .on_click(move |_, window, cx| {
                    let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
                    run(window, cx);
                })
        });
        let me = cx.entity().downgrade();
        div()
            .id("palette-backdrop")
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .flex()
            .justify_center()
            .pt(px(90.))
            .bg(gpui_kit::hsla(0., 0., 0., 0.4))
            .on_click(move |_, _, cx| {
                let _ = me.update(cx, |_, cx| cx.emit(PaletteEvent::Close));
            })
            .on_key_down(cx.listener(|_, e: &gpui_kit::KeyDownEvent, _, cx| {
                if e.keystroke.key == "escape" {
                    cx.emit(PaletteEvent::Close);
                }
            }))
            .child(
                v_flex()
                    .id("palette")
                    .w(px(600.))
                    .max_h(px(540.))
                    .p(px(theme::SM))
                    .gap(px(theme::XS))
                    .rounded(px(8.))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .shadow_lg()
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .child(Input::new(&self.input))
                    .child(v_flex().id("palette-rows").overflow_y_scroll().children(rows))
                    .child(div().px(px(theme::XS)).pt(px(theme::XS)).text_xs().text_color(color::text_muted()).child("Enter runs the first · Esc closes")),
            )
    }
}

/// Highlights the first row (it runs on Enter).
trait FirstRow: Sized {
    fn when_first(self, first: bool) -> Self;
}

impl FirstRow for gpui_kit::Stateful<gpui_kit::Div> {
    fn when_first(self, first: bool) -> Self {
        if first { self.bg(color::control()) } else { self }
    }
}
