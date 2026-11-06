//! The Notes tab: the project's notes; a new one is pinned to the centre of the view, and a click
//! on a pinned note goes back there.

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Icon, Sizable as _};
use gpui_kit::{div, px, AppContext as _, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Subscription, Window};

use crate::data::store::{AppEvent, Store};
use crate::domain::command::Command;
use crate::domain::project::NoteDoc;
use crate::screens::explore::canvas::ImageCanvas;
use crate::ui::theme::{self, color};
use crate::ui::widgets::Section;

pub struct NotesPanel {
    store: Entity<Store>,
    canvas: Entity<ImageCanvas>,
    input: Entity<InputState>,
    _subs: Vec<Subscription>,
}

impl NotesPanel {
    pub fn new(store: Entity<Store>, canvas: Entity<ImageCanvas>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let input = cx.new(|cx| InputState::new(window, cx).placeholder("Write a note, Enter adds it…"));
        let subs = vec![
            cx.subscribe(&store, |_, _, e: &AppEvent, cx| {
                if matches!(e, AppEvent::Document(_)) {
                    cx.notify();
                }
            }),
            cx.subscribe_in(&input, window, |this, input, e: &InputEvent, window, cx| {
                if matches!(e, InputEvent::PressEnter { .. }) {
                    this.add(input, window, cx);
                }
            }),
        ];
        Self { store, canvas, input, _subs: subs }
    }

    fn add(&mut self, input: &Entity<InputState>, window: &mut Window, cx: &mut Context<Self>) {
        let text = input.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        let image = self.store.read(cx).doc.active_image.clone();
        let c = self.canvas.read(cx).center;
        let note = NoteDoc { id: String::new(), text, at: image.as_ref().map(|_| [c.0, c.1]), image };
        if self.store.update(cx, |s, cx| s.execute(Command::SetNote { note }, cx)).is_ok() {
            input.update(cx, |i, cx| i.set_value("", window, cx));
        }
    }
}

impl Render for NotesPanel {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let notes = self.store.read(cx).doc.notes.clone();
        let mut list = v_flex().w_full().gap(px(theme::XS));
        for n in notes.iter().rev() {
            let (canvas, store, id, at) = (self.canvas.clone(), self.store.clone(), n.id.clone(), n.at);
            list = list.child(
                h_flex()
                    .id(SharedString::from(format!("note-{}", n.id)))
                    .w_full()
                    .items_start()
                    .gap(px(theme::SM))
                    .p(px(theme::SM))
                    .rounded(px(5.))
                    .bg(color::raised())
                    .hover(|d| d.bg(color::control()))
                    .cursor_pointer()
                    .child(Icon::new(if at.is_some() { IconName::MapPin } else { IconName::StickyNote }).small().text_color(color::text_muted()))
                    .child(div().flex_1().min_w_0().text_sm().child(n.text.clone()))
                    .child(Button::new(SharedString::from(format!("note-del-{}", n.id))).ghost().xsmall().icon(Icon::new(IconName::X)).on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        store.update(cx, |s, cx| {
                            let _ = s.execute(Command::RemoveNote { id: id.clone() }, cx);
                        });
                    }))
                    .on_click(move |_, _, cx| {
                        if let Some(p) = at {
                            canvas.update(cx, |c, cx| c.reveal(p, cx));
                        }
                    }),
            );
        }
        v_flex()
            .id("notes-panel")
            .size_full()
            .overflow_y_scroll()
            .p(px(theme::MD))
            .gap(px(theme::MD))
            .child(Section::new("Notes").child(Input::new(&self.input)))
            .child(list)
    }
}
