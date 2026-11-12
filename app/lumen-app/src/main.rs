//! Lumen: GPU workstation for histology and radiology (GPUI). The app is `lumen-app-core`.

use gpui_kit::component::TitleBar;
use gpui_kit::{px, size, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions};
use lumen_app_core::app::LumenApp;
use lumen_app_core::data::prefs::Prefs;
use lumen_app_core::data::store::Store;
use lumen_app_core::navigation::actions;

fn main() {
    // An agent's MCP server: the running app's commands, through its socket
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--mcp-bridge") {
        let Some(socket) = args.get(2) else {
            eprintln!("usage: lumen-app --mcp-bridge <socket>");
            std::process::exit(2);
        };
        if let Err(e) = kitchen_agent::bridge::bridge_stdio(std::path::Path::new(socket)) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    // `lumen-app <path>` opens an image, an OME-Zarr container or a project
    let open = std::env::args().nth(1).map(std::path::PathBuf::from);
    gpui_kit::application().with_assets(kitchen_ui::icons::AppAssets::new().all_icons()).run(move |cx| {
        gpui_kit::init(cx);
        actions::bind_keys(cx);
        cx.on_window_closed(|cx, _| cx.quit()).detach();
        let prefs_path = Prefs::path();
        let store = cx.new(|_| Store::new(Prefs::load(prefs_path.as_deref()), prefs_path));
        let options = WindowOptions {
            titlebar: Some(TitlebarOptions { title: Some("Lumen".into()), ..TitleBar::title_bar_options() }),
            window_bounds: Some(WindowBounds::Windowed(Bounds::centered(None, size(px(1480.), px(920.)), cx))),
            window_min_size: Some(size(px(900.), px(560.))),
            app_id: Some("org.neuro-kitchen.lumen".into()),
            // The app draws its own frame on Linux (rounded corners and a shadow while it floats):
            // the window behind them is transparent
            window_background: gpui_kit::WindowBackgroundAppearance::Transparent,
            window_decorations: Some(gpui_kit::WindowDecorations::Client),
            ..TitleBar::window_options()
        };
        let s = store.clone();
        gpui_kit::open_window(options, cx, |window, cx| cx.new(|cx| LumenApp::new(s, window, cx))).expect("failed to open the main window");
        cx.activate(true);
        if let Some(p) = open.clone() {
            let p = std::fs::canonicalize(&p).unwrap_or(p);
            store.update(cx, |s, cx| s.open(p, cx));
        }
    });
}
