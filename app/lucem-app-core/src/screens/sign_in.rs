//! Signing in to a lucem server with a pairing code, in the app's own screen (not the browser's
//! prompt): where the code comes from, the field, and what went wrong if it did. Shown over the
//! frame (the shell's overlay) by whoever connects (the browser page); the code goes to its
//! callback, which answers with an error to show, or closes the form.

use std::rc::Rc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputEvent, InputState};
use gpui_kit::component::{h_flex, v_flex, Disableable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, Styled as _, Subscription, Window};

use crate::ui::theme::{self, color};

/// What the form does with a code typed in.
pub type OnCode = Rc<dyn Fn(String, &mut Window, &mut gpui_kit::App)>;

pub struct SignIn {
    server: SharedString,
    code: Entity<InputState>,
    error: Option<SharedString>,
    busy: bool,
    on_code: OnCode,
    _subs: Vec<Subscription>,
}

impl SignIn {
    /// A form for the server named `server`; `on_code` gets the code typed in.
    pub fn new(server: impl Into<SharedString>, on_code: OnCode, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let code = cx.new(|cx| InputState::new(window, cx).placeholder("42-8317-2956"));
        let subs = vec![cx.subscribe_in(&code, window, |this, _, e: &InputEvent, window, cx| {
            if matches!(e, InputEvent::PressEnter { .. }) {
                this.submit(window, cx);
            }
        })];
        code.update(cx, |c, cx| c.focus(window, cx));
        Self { server: server.into(), code, error: None, busy: false, on_code, _subs: subs }
    }

    /// Says why the last code did not work (the form stays open for another).
    pub fn refused(&mut self, why: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.error = Some(why.into());
        self.busy = false;
        cx.notify();
    }

    fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let code = self.code.read(cx).value().to_string();
        let digits = code.chars().filter(char::is_ascii_digit).count();
        if digits != 10 {
            self.error = Some("A pairing code is ten digits, like 42-8317-2956".into());
            cx.notify();
            return;
        }
        self.error = None;
        self.busy = true;
        cx.notify();
        let on_code = self.on_code.clone();
        on_code(code, window, cx);
    }
}

impl Render for SignIn {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("sign-in-overlay")
            .absolute()
            // Takes the pointer: a click here never reaches the buttons under it
            .occlude()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui_kit::hsla(0., 0., 0., 0.55))
            .child(
                v_flex()
                    .id("sign-in")
                    .w(px(440.))
                    .max_w_full()
                    // Padding at least 1.5× the gap between its parts (`padding-gap-ratio`)
                    .p(px(theme::XL))
                    .gap(px(theme::MD))
                    .rounded(px(theme::RADIUS_LG))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child(format!("Sign in to {}", self.server)))
                    .child(div().text_sm().text_color(color::text_muted()).child("Pair this browser once with a code from the server: in its log (lucem serve --pair), or made with lucem clients pair. The browser keeps a sign-in of its own after that; the server can revoke it."))
                    .child(v_flex().gap(px(theme::XS)).child(div().text_xs().font_weight(FontWeight::SEMIBOLD).child("Pairing code")).child(Input::new(&self.code)))
                    .when_some(self.error.clone(), |d, e| d.child(div().text_sm().text_color(color::danger()).child(e)))
                    .child(
                        h_flex().justify_end().child(
                            Button::new("sign-in-pair")
                                .primary()
                                .label(if self.busy { "Pairing…" } else { "Pair this browser" })
                                .disabled(self.busy)
                                .on_click(cx.listener(|this, _, window, cx| this.submit(window, cx))),
                        ),
                    ),
            )
    }
}
