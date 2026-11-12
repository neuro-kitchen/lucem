//! Headless UI tests (dsp-kitchen's harness): the real app in a test window, a built-in sample
//! image (no data files), every frame laid out and hit-tested on the way.

use std::time::Duration;

use gpui_kit::test::TestWindowExt as _;
use gpui_kit::{px, size, AnyWindowHandle, AppContext as _, Bounds, Entity, Point, TestAppContext, WindowBounds, WindowOptions};

use super::LumenApp;
use crate::data::prefs::Prefs;
use crate::data::store::{Arrangement, Store, ViewKind};
use crate::navigation::route::Workspace;

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

/// Closes whatever a command opened (Settings, the search, the help sheet), so the next starts
/// from the plain app.
fn reset(cx: &mut TestAppContext, window: AnyWindowHandle, app: &Entity<LumenApp>) {
    let shell = cx.update(|cx| app.read(cx).shell.clone());
    cx.update_window(window, |_, window, cx| {
        shell.update(cx, |s, cx| {
            s.show_settings(None, window, cx);
            s.set_overlay(None, window, cx);
            if s.search().is_some() {
                s.open_search(window, cx);
            }
        })
    })
    .unwrap();
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

/// Every command runs and the app draws after each (those that open a system file dialog or
/// quit aside; those that cannot run now say why, and those needing arguments say which).
#[gpui_kit::test]
fn every_command_runs(cx: &mut TestAppContext) {
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    let registry = cx.update(|cx| app.read(cx).registry.clone());
    let commands = registry.all();
    assert!(commands.len() > 100, "every action is a command: {}", commands.len());
    // File dialogs, quitting, and the exports (they write files where the test runs)
    let dialog = ["file.open_image", "file.open_project", "file.save_in", "file.import_geojson", "script.from_file", "app.quit", "file.new", "image.remove", "file.export_geojson", "file.export_cells", "analyze.export_csv", "agent.start", "agent.use"];
    for c in commands.into_iter().filter(|c| !dialog.contains(&c.id.as_ref())) {
        let r = cx.update_window(window, |_, window, cx| registry.run(&c.id, serde_json::Value::Null, window, cx)).unwrap();
        if let Err(e) = r {
            assert!(!e.is_empty(), "{} says why it did not run", c.id);
        }
        frame(cx, window);
        reset(cx, window, &app);
    }
}

/// Every menu of every workspace names registered commands, its providers' entries too.
#[gpui_kit::test]
fn every_menu_entry_is_a_command(cx: &mut TestAppContext) {
    use kitchen_shell::MenuItem;
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    let registry = cx.update(|cx| app.read(cx).registry.clone());
    fn walk(items: &[MenuItem], registry: &kitchen_shell::Registry, cx: &gpui_kit::App, out: &mut Vec<String>) {
        for i in items {
            match i {
                MenuItem::Command(id) | MenuItem::Run { command: id, .. } => out.push(id.to_string()),
                MenuItem::Submenu { items, .. } => walk(items, registry, cx, out),
                MenuItem::Provided(p) => walk(&registry.provided(p, cx), registry, cx, out),
                _ => {}
            }
        }
    }
    let mut menus = crate::commands::explore_menus();
    menus.extend(crate::screens::pipeline::PipelineView::menu_specs());
    menus.extend(crate::screens::analyze::AnalyzeView::menu_specs());
    let mut ids = Vec::new();
    cx.update(|cx| {
        for m in &menus {
            walk(&m.items, &registry, cx, &mut ids);
        }
        walk(&crate::commands::app_menu(), &registry, cx, &mut ids);
    });
    assert!(ids.len() > 80, "the menus hold the commands: {}", ids.len());
    for id in ids {
        assert!(registry.get(&id).is_some(), "{id} is registered");
    }
}

/// Settings chosen in the search opens (the command runs once the search has closed).
#[gpui_kit::test]
fn settings_opens_from_the_search(cx: &mut TestAppContext) {
    let (window, _store, app) = open(cx);
    let shell = cx.update(|cx| app.read(cx).shell.clone());
    cx.update_window(window, |_, window, cx| shell.update(cx, |s, cx| s.open_search(window, cx))).unwrap();
    let search = cx.update(|cx| shell.read(cx).search().expect("the search is open"));
    cx.update(|cx| search.update(cx, |_, cx| cx.emit(kitchen_shell::SearchEvent::Run("shell.settings".into()))));
    frame(cx, window);
    frame(cx, window);
    cx.update(|cx| {
        assert!(shell.read(cx).search().is_none(), "the search closed");
        assert_eq!(shell.read(cx).settings_section(), Some(0), "Settings opened");
    });
    // lumen's sections are there, after the frame's Appearance
    for (i, sec) in crate::screens::settings::Section::ALL.iter().enumerate() {
        assert_eq!(cx.update(|cx| shell.read(cx).settings_index(sec.title())), Some(i + 1));
        cx.update_window(window, |_, window, cx| shell.update(cx, |s, cx| s.show_settings(Some(i + 1), window, cx))).unwrap();
        frame(cx, window);
    }
}

/// The workspace follows the store and the frame alike.
#[gpui_kit::test]
fn the_frame_and_the_store_share_the_workspace(cx: &mut TestAppContext) {
    let (window, store, app) = open(cx);
    let shell = cx.update(|cx| app.read(cx).shell.clone());
    cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx)));
    frame(cx, window);
    assert_eq!(cx.update(|cx| shell.read(cx).workspace_id()).as_ref(), "analyze");
    cx.update(|cx| shell.update(cx, |s, cx| s.set_workspace("pipeline", cx)));
    frame(cx, window);
    assert_eq!(cx.update(|cx| store.read(cx).workspace), Workspace::Pipeline);
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

/// What you do lands on the activity timeline: commands run show in Analyze's Activity view, on
/// your lane, with the workspace and how they were started.
#[gpui_kit::test]
fn the_activity_view_shows_what_was_done(cx: &mut TestAppContext) {
    let (window, store, app) = open(cx);
    open_sample(cx, window, &store);
    let registry = cx.update(|cx| app.read(cx).registry.clone());
    cx.update_window(window, |_, window, cx| {
        let _ = registry.run_as(kitchen_shell::activity::Actor::Person, "menu", "view.rulers", serde_json::Value::Null, window, cx);
        let _ = registry.run_as(kitchen_shell::activity::Actor::Agent, "agent", "hist.find_tissue", serde_json::Value::Null, window, cx);
    })
    .unwrap();
    cx.update(|cx| store.update(cx, |s, cx| s.set_workspace(Workspace::Analyze, cx)));
    frame(cx, window);
    let analyze = cx.update(|cx| app.read(cx).analyze.clone());
    cx.update_window(window, |_, window, cx| analyze.update(cx, |a, cx| a.add_activity(window, cx))).unwrap();
    frame(cx, window);
    let timeline = cx.update(|cx| analyze.read(cx).activity.clone().expect("the timeline is made"));
    let acts = cx.update(|cx| timeline.read(cx).activities().to_vec());
    let rulers = acts.iter().find(|a| a.title.as_ref() == "Rulers").expect("the rulers command is on the timeline");
    assert_eq!(rulers.lane.as_ref(), "You");
    assert!(rulers.source.contains("menu"), "{}", rulers.source);
    assert!(acts.iter().any(|a| a.lane.as_ref() == "Agent" && a.title.as_ref() == "Find tissue"));
    assert!(acts.iter().any(|a| a.lane.as_ref() == "Focus"), "time in each workspace");
    assert!(cx.update(|cx| analyze.read(cx).activity_panel.is_some()));
}
