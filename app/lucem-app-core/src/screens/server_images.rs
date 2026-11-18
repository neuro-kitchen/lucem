//! The images of the server, to open one: a browser page's "Open image…" (it has no file
//! dialog; it views what its server serves), as dsp-kitchen's "Open from server". Shown over the
//! frame; asks the server when it opens, then lists what it said.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{h_flex, v_flex, Sizable as _};
use gpui_kit::{div, px, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Window};

use crate::data::store::Store;
use crate::ui::theme::{self, color};

/// Closes the list (the frame's overlay goes away).
pub type Close = Rc<dyn Fn(&mut Window, &mut App)>;

pub struct ServerImages {
    store: Entity<Store>,
    close: Close,
    /// By server address: its images, or why it could not say; `None` while asking.
    listed: Option<Vec<(String, Result<Vec<String>, String>)>>,
}

impl ServerImages {
    /// The list, asking the servers now (in the background: a page waits on a worker).
    pub fn new(store: Entity<Store>, close: Close, cx: &mut Context<Self>) -> Self {
        let task = cx.background_spawn(async {
            crate::data::remote::refresh();
            crate::data::remote::last_listed().into_iter().collect::<Vec<_>>()
        });
        cx.spawn(async move |this, cx| {
            let listed = task.await;
            let _ = this.update(cx, |this, cx| {
                this.listed = Some(listed);
                cx.notify();
            });
        })
        .detach();
        Self { store, close, listed: None }
    }

    fn open(&self, addr: &str, path: &str, window: &mut Window, cx: &mut App) {
        let name = lucem_stream::remote_name(addr, path);
        self.store.update(cx, |s, cx| s.open(std::path::PathBuf::from(name), cx));
        (self.close)(window, cx);
    }
}

impl Render for ServerImages {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = color::text_muted();
        let mut list = v_flex().gap(px(theme::XS));
        match &self.listed {
            None => list = list.child(div().text_sm().text_color(muted).child("Asking the server…")),
            Some(servers) if servers.is_empty() => list = list.child(div().text_sm().text_color(muted).child("No server is connected")),
            Some(servers) => {
                for (addr, found) in servers.clone() {
                    match found {
                        Ok(images) if images.is_empty() => list = list.child(div().text_sm().text_color(muted).child(format!("{addr}: no images in its data folder"))),
                        Ok(images) => {
                            for (i, path) in images.into_iter().enumerate() {
                                let (a, p) = (addr.clone(), path.clone());
                                list = list.child(
                                    Button::new(SharedString::from(format!("server-image-{i}")))
                                        .ghost()
                                        .icon(IconName::Image)
                                        .label(path)
                                        .on_click(cx.listener(move |this, _, window, cx| this.open(&a, &p, window, cx))),
                                );
                            }
                        }
                        Err(e) => list = list.child(div().text_sm().text_color(color::danger()).child(e)),
                    }
                }
            }
        }
        let close = self.close.clone();
        div()
            .id("server-images-overlay")
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
                    .id("server-images")
                    .w(px(520.))
                    .max_w_full()
                    .max_h(px(560.))
                    // Padding at least 1.5× the gap between its parts (`padding-gap-ratio`)
                    .p(px(theme::XL))
                    .gap(px(theme::MD))
                    .rounded(px(theme::RADIUS_LG))
                    .border_1()
                    .border_color(color::border())
                    .bg(color::raised())
                    .child(div().text_lg().font_weight(FontWeight::SEMIBOLD).child("Open an image"))
                    .child(div().text_sm().text_color(muted).child("On a server: the images under its data folder; in this browser alone: the ones among the files you picked. Nothing is uploaded: tiles are read where the image is."))
                    .child(div().id("server-images-list").flex_1().overflow_y_scroll().child(list))
                    .child(h_flex().justify_end().child(Button::new("server-images-close").outline().small().label("Close").on_click(move |_, w, cx| close(w, cx)))),
            )
    }
}
