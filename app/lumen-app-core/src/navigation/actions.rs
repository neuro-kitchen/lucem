//! App actions and their key bindings (`secondary` = Cmd on macOS, Ctrl elsewhere). Tool keys
//! (V, H, B, …) are handled by the image canvas while it has focus, so typing never switches tools.

use gpui_kit::{actions, Action, App, KeyBinding};

actions!(
    lumen_app,
    [
        /// Choose an image or OME-Zarr container and open it.
        OpenImage,
        OpenProject,
        SaveProjectAs,
        NewProject,
        Undo,
        Redo,
        CommandPalette,
        ShowHelp,
        Quit,
        GoExplore,
        GoPipeline,
        GoAnalyze,
        ToggleSidebar,
        ToggleRulers,
        OpenSettings,
        ToggleLeftPanel,
        ToggleRightPanel,
    ]
);

/// A global binding whose `secondary-` modifier is resolved for this OS.
fn os_binding<A: Action>(keys: &str, action: A, cx: &App) -> KeyBinding {
    KeyBinding::load(keys, Box::new(action), None, false, None, cx).unwrap_or_else(|e| panic!("key binding {keys:?}: {e:?}"))
}

pub fn bind_keys(cx: &mut App) {
    let global = [
        os_binding("secondary-o", OpenImage, cx),
        os_binding("secondary-shift-o", OpenProject, cx),
        os_binding("secondary-shift-s", SaveProjectAs, cx),
        os_binding("secondary-n", NewProject, cx),
        os_binding("secondary-z", Undo, cx),
        os_binding("secondary-shift-z", Redo, cx),
        os_binding("secondary-y", Redo, cx),
        os_binding("secondary-k", CommandPalette, cx),
        os_binding("secondary-q", Quit, cx),
        os_binding("secondary-1", GoExplore, cx),
        os_binding("secondary-2", GoPipeline, cx),
        os_binding("secondary-3", GoAnalyze, cx),
        os_binding("secondary-r", ToggleRulers, cx),
        os_binding("secondary-,", OpenSettings, cx),
        os_binding("secondary-[", ToggleLeftPanel, cx),
        os_binding("secondary-]", ToggleRightPanel, cx),
    ];
    cx.bind_keys(global);
    cx.bind_keys([KeyBinding::new("f1", ShowHelp, None), KeyBinding::new("f3", CommandPalette, None), KeyBinding::new("tab", ToggleSidebar, Some("ImageCanvas"))]);
}

/// Every binding as (keys, what it does), for the Keyboard & mouse dialog.
pub const HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("Ctrl+O / Ctrl+Shift+O", "Open an image / a project"),
            ("Ctrl+Shift+S", "Save the project in a folder"),
            ("Ctrl+K · F3", "Search: any operation, tool or action by name"),
            ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo any change"),
            ("Ctrl+1 / 2 / 3", "Explore · Pipeline · Analyze"),
            ("Ctrl+R", "Rulers on / off"),
            ("Ctrl+,", "Settings (theme, accent, sizes)"),
            ("Ctrl+[ / Ctrl+]", "Show / hide the left (Layers) / right (studio) panels"),
            ("F1", "This list"),
        ],
    ),
    (
        "On the image (click it first)",
        &[
            ("Wheel · pinch", "Zoom at the pointer"),
            ("Drag (Hand) · Space+drag · middle drag", "Move the image"),
            ("Two-finger swipe", "Move the image"),
            ("F · Ctrl+0 / Ctrl+1", "Fit · 100 %"),
            ("V H Z", "Select · Hand · Zoom"),
            ("M O P L N", "Rectangle · Ellipse · Polygon · Lasso · Point"),
            ("B E · [ ]", "Brush · Eraser · brush size"),
            ("W T", "Wand (snaps to similar pixels) · Threshold (rapid segmentation)"),
            ("R I", "Ruler · Pixel probe"),
            ("Enter · Esc · Backspace", "Close the polygon · cancel · remove its last point"),
            ("Delete", "Delete the selected annotations"),
            ("Tab", "Hide / show both side panels"),
            ("Drag a number", "Any value with − and + (and every %) changes as you drag it left or right"),
        ],
    ),
];
