//! Headless UI tests (dsp-kitchen's harness): the real app in a test window, a built-in sample
//! image (no data files), every frame laid out and hit-tested on the way.

use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{px, size, AnyWindowHandle, AppContext as _, Bounds, Entity, Point, TestAppContext, WindowBounds, WindowOptions};

use super::LumenApp;
use crate::data::prefs::Prefs;
use crate::data::store::{Arrangement, Store, ViewKind};
use crate::navigation::route::Workspace;
use crate::screens::palette::PaletteEvent;

/// Opens the app (default preferences, never saved) in a 1400 × 900 test window.
fn open(cx: &mut TestAppContext) -> (AnyWindowHandle, Entity<Store>, Entity<LumenApp>) {
    cx.executor().allow_parking();
    cx.update(|cx| {
        gpui_kit::init(cx);
        crate::navigation::actions::bind_keys(cx);
        let store = cx.new(|_| Store::new(Prefs::default(), None));
        let bounds = Bounds { origin: Point::default(), size: size(px(1400.), px(900.)) };
        let options = WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        let s = store.clone();
        let (window, app) = gpui_kit::open_window(options, cx, move |window, cx| cx.new(|cx| LumenApp::new(s, window, cx))).expect("open test window");
        (window, store, app)
    })
}

/// Draws a frame and lets the UI settle.
fn frame(cx: &mut TestAppContext, window: AnyWindowHandle) {
    cx.update_window(window, |_, window, cx| window.render_frame(cx)).unwrap();
    cx.run_until_parked();
}

/// Waits (real time) until `done`, drawing frames meanwhile.
fn wait_until(cx: &mut TestAppContext, window: AnyWindowHandle, what: &str, mut done: impl FnMut(&mut TestAppContext) -> bool) {
    for _ in 0..1000 {
        frame(cx, window);
        if done(cx) {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

/// Opens the built-in sample volume and waits until it is shown.
fn open_sample(cx: &mut TestAppContext, window: AnyWindowHandle, store: &Entity<Store>) {
    cx.update(|cx| store.update(cx, |s, cx| s.open("sample:volume".into(), cx)));
    wait_until(cx, window, "the sample image", |cx| cx.update(|cx| store.read(cx).active_pyramid().is_some()));
}

/// Closes whatever a command opened, so the next starts from the plain app.
fn reset(cx: &mut TestAppContext, app: &Entity<LumenApp>) {
    cx.update(|cx| {
        app.update(cx, |a, cx| {
            a.settings = None;
            a.show_help = false;
            a.palette = None;
            cx.notify();
        })
    });
}

/// The app opens, and every workspace draws, without and with an image.
#[gpui_kit::test]
fn every_workspace_draws(cx: &mut TestAppContext) {
    let (window, store, _app) = open(cx);
    for w in Workspace::ALL {
        cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(w, cx)));
        frame(cx, window);
    }
    open_sample(cx, window, &store);
    for w in Workspace::ALL {
        cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(w, cx)));
        frame(cx, window);
    }
}

/// Every search command runs and the app draws after each (those that open a system file
/// dialog, quit or restart aside).
#[gpui_kit::test]
fn every_search_command_runs(cx: &mut TestAppContext) {
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    let commands = cx.update_window(window, |_, _, cx| app.update(cx, |a, cx| a.commands(false, cx))).unwrap();
    assert!(commands.len() > 40, "the search lists every action: {}", commands.len());
    let dialog = |name: &str| ["Open", "Import", "Save", "Quit", "Restart", "New project"].iter().any(|w| name.starts_with(w));
    for c in commands.into_iter().filter(|c| !dialog(&c.name)) {
        let run = c.run.clone();
        cx.update_window(window, |_, window, cx| run(window, cx)).unwrap();
        frame(cx, window);
        reset(cx, &app);
    }
}

/// Settings chosen in the search opens (the command runs once the search has closed).
#[gpui_kit::test]
fn settings_opens_from_the_search(cx: &mut TestAppContext) {
    let (window, _store, app) = open(cx);
    cx.update_window(window, |_, window, cx| app.update(cx, |a, cx| a.open_search(false, window, cx))).unwrap();
    let palette = cx.update(|cx| app.read(cx).palette.as_ref().map(|(p, _)| p.clone()).expect("the search is open"));
    let settings = cx.update_window(window, |_, _, cx| app.update(cx, |a, cx| a.commands(false, cx))).unwrap().into_iter().find(|c| c.name.starts_with("Settings")).expect("Settings is listed").run;
    cx.update(|cx| palette.update(cx, |_, cx| cx.emit(PaletteEvent::Run(settings))));
    frame(cx, window);
    frame(cx, window);
    cx.update(|cx| {
        assert!(app.read(cx).palette.is_none(), "the search closed");
        assert!(app.read(cx).settings.is_some(), "Settings opened");
    });
}

/// Explore: a view of every kind is added, then every arrangement is applied.
#[gpui_kit::test]
fn explore_views_add_and_arrange(cx: &mut TestAppContext) {
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    let explore = cx.update(|cx| app.read(cx).explore.clone());
    for kind in ViewKind::ALL {
        cx.update_window(window, |_, window, cx| explore.update(cx, |e, cx| e.add_view(kind, window, cx))).unwrap();
        frame(cx, window);
    }
    assert_eq!(cx.update(|cx| explore.read(cx).panels.len()), 1 + ViewKind::ALL.len());
    for how in Arrangement::ALL {
        cx.update_window(window, |_, window, cx| explore.update(cx, |e, cx| e.arrange(how, window, cx))).unwrap();
        frame(cx, window);
    }
}

/// Analyze: every plot kind is added beside the table, every arrangement applied, the table
/// closed and brought back.
#[gpui_kit::test]
fn analyze_views_add_arrange_and_close(cx: &mut TestAppContext) {
    use crate::screens::analyze::{PanelKind, Plot};
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx)));
    frame(cx, window);
    let analyze = cx.update(|cx| app.read(cx).analyze.clone());
    for kind in [Plot::Histogram, Plot::Scatter, Plot::Box, Plot::Counts] {
        cx.update_window(window, |_, window, cx| analyze.update(cx, |a, cx| a.add_plot(kind, window, cx))).unwrap();
        frame(cx, window);
    }
    assert_eq!(cx.update(|cx| analyze.read(cx).plot_panels.len()), 4);
    for how in Arrangement::ALL {
        cx.update_window(window, |_, window, cx| analyze.update(cx, |a, cx| a.arrange(how, window, cx))).unwrap();
        frame(cx, window);
    }
    // The table leaves the dock and comes back
    let table = cx.update(|cx| analyze.read(cx).table_panel.clone().expect("the table is shown"));
    cx.update_window(window, |_, window, cx| {
        let dock = analyze.read(cx).dock.clone();
        dock.update(cx, |d, cx| d.remove_panel(table.clone(), window, cx));
        analyze.update(cx, |a, cx| a.removed(PanelKind::Table, cx));
    })
    .unwrap();
    frame(cx, window);
    assert!(cx.update(|cx| analyze.read(cx).table_panel.is_none()));
    cx.update_window(window, |_, window, cx| analyze.update(cx, |a, cx| a.add_table(window, cx))).unwrap();
    frame(cx, window);
    assert!(cx.update(|cx| analyze.read(cx).table_panel.is_some()));
}

/// A project saved in a folder writes its graph and its views beside it (`pipeline.json`,
/// `layout.json`); opened again in a new app, Explore's views and Analyze's plots come back.
#[gpui_kit::test]
fn a_saved_project_brings_back_its_views(cx: &mut TestAppContext) {
    use crate::screens::analyze::Plot;
    let folder = std::env::temp_dir().join(format!("lumen-ui-project-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    {
        let (window, store, app) = open(cx);
        open_sample(cx, window, &store);
        let explore = cx.update(|cx| app.read(cx).explore.clone());
        cx.update_window(window, |_, window, cx| explore.update(cx, |e, cx| e.add_view(ViewKind::Volume, window, cx))).unwrap();
        cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx)));
        frame(cx, window);
        let analyze = cx.update(|cx| app.read(cx).analyze.clone());
        cx.update_window(window, |_, window, cx| analyze.update(cx, |a, cx| a.add_plot(Plot::Scatter, window, cx))).unwrap();
        frame(cx, window);
        cx.update(|cx| store.update(cx, |s, cx| s.save_in(folder.clone(), cx)));
        frame(cx, window);
    }
    for f in [crate::domain::project::PROJECT_FILE, crate::domain::project::PIPELINE_FILE, crate::data::layout::LAYOUT_FILE] {
        assert!(folder.join(f).is_file(), "{f} written");
    }
    let layout = std::fs::read_to_string(folder.join(crate::data::layout::LAYOUT_FILE)).unwrap();
    assert!(layout.contains("3D") && layout.contains("Scatter"), "the views are saved: {layout}");

    let (window, store, app) = open(cx);
    cx.update(|cx| store.update(cx, |s, cx| s.open(folder.join(crate::domain::project::PROJECT_FILE), cx)));
    for _ in 0..5 {
        frame(cx, window);
    }
    cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx)));
    for _ in 0..3 {
        frame(cx, window);
    }
    let explore = cx.update(|cx| app.read(cx).explore.clone());
    let analyze = cx.update(|cx| app.read(cx).analyze.clone());
    cx.update(|cx| {
        let kinds: Vec<ViewKind> = explore.read(cx).panels.iter().map(|p| p.read(cx).kind).collect();
        assert_eq!(kinds, vec![ViewKind::Plane, ViewKind::Volume], "Explore's views");
        assert_eq!(analyze.read(cx).plots.len(), 1, "Analyze's plot");
    });
    std::fs::remove_dir_all(&folder).ok();
}
