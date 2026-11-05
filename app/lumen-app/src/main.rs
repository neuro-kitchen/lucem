//! Entry point for `lumen-app`: launches the GPUI window with ArtCraft × VectorCraft × Blender
//! dark studio chrome, CubeCL GPU runtime, and Polars/Arrow tabular analytics.

pub mod app;
pub mod data;
pub mod di;
pub mod domain;
pub mod navigation;
pub mod screens;
pub mod ui;

use app::LumenApp;
use gpui_kit::{px, size, AppContext, Bounds, TitlebarOptions, WindowBounds, WindowOptions};

fn main() {
    gpui_kit::application()
        .with_assets(gpui_kit::assets::AllAssets)
        .run(|cx| {
            gpui_kit::init(cx);

            let bounds = Bounds::centered(None, size(px(1440.0), px(900.0)), cx);
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("Lumen Studio — ArtCraft × VectorCraft × Blender GPU Lab".into()),
                    appears_transparent: false,
                    traffic_light_position: None,
                }),
                ..WindowOptions::default()
            };

            gpui_kit::open_window(options, cx, |window, cx| {
                cx.new(|cx| LumenApp::new(window, cx))
            })
            .expect("failed to open Lumen Studio window");
        });
}
