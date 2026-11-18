//! Lucem: GPU workstation for histology and radiology (GPUI). The app is `lucem-app-core`.

use gpui_kit::component::TitleBar;
use gpui_kit::{px, size, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions};
use lucem_app_core::app::LucemApp;
use lucem_app_core::data::prefs::Prefs;
use lucem_app_core::data::store::Store;
use lucem_app_core::navigation::actions;

fn main() {
    // Profiling: every span goes to Tracy while it runs
    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt as _;
        let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()));
    }
    lucem_app_core::legacy::adopt();
    // `lucem-app <path>` opens an image, an OME-Zarr container or a project
    let open = std::env::args().nth(1).map(std::path::PathBuf::from);
    gpui_kit::application().with_assets(lucem_app_core::ui::assets::app_assets()).run(move |cx| {
        gpui_kit::init(cx);
        actions::bind_keys(cx);
        cx.on_window_closed(|cx, _| cx.quit()).detach();
        let prefs_path = Prefs::path();
        let store = cx.new(|_| Store::new(Prefs::load(prefs_path.as_deref()), prefs_path));
        let options = WindowOptions {
            titlebar: Some(TitlebarOptions { title: Some("Lucem".into()), ..TitleBar::title_bar_options() }),
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(None, size(px(1480.), px(920.)), cx))),
            window_min_size: Some(size(px(900.), px(560.))),
            app_id: Some("org.neuro-kitchen.lucem".into()),
            // The app draws its own frame on Linux (rounded corners and a shadow while it floats):
            // the window behind them is transparent
            window_background: gpui_kit::WindowBackgroundAppearance::Transparent,
            window_decorations: Some(gpui_kit::WindowDecorations::Client),
            ..TitleBar::window_options()
        };
        let s = store.clone();
        gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| LucemApp::new(s, window, cx))).expect("failed to open the main window");
        cx.activate(true);
        match open.clone() {
            Some(p) => {
                let p = std::fs::canonicalize(&p).unwrap_or(p);
                store.update(cx, |s, cx| s.open(p, cx));
            }
            // The work of the last run, as it was left
            None => {
                store.update(cx, |s, cx| s.restore_session(cx));
            }
        }
    });
}
