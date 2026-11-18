//! Keys that only work on the image canvas (its key context). Every other key belongs to a
//! command (`crate::commands`), bound by kitchen-shell's registry. Tool keys (V, H, B, …) are
//! handled by the canvas while it has focus, so typing never switches tools.

use gpui_kit::{actions, App, KeyBinding};

actions!(
    lucem_app,
    [
        /// Hide or show both side panels (Tab on the image).
        ToggleSidebar,
    ]
);

/// The canvas' commands and their default keys: the `[canvas]` section of `keybindings.toml`.
pub const CANVAS_KEYS: &[(&str, &[&str])] = &[("toggle_sidebar", &["tab"])];

/// Reads `keybindings.toml` (once, at start) and binds the canvas' own keys: the defaults, or
/// what the file says in `[canvas]`. The commands' keys are bound with the frame (`[commands]`).
pub fn bind_keys(cx: &mut App) {
    if let Some(path) = crate::data::prefs::Prefs::keybindings_path() {
        kitchen_shell::keys::load_keybindings(&path);
    }
    let context = gpui_kit::KeyBindingContextPredicate::parse("ImageCanvas").ok().map(std::rc::Rc::new);
    let load = |key: &str, cx: &App| KeyBinding::load(key, Box::new(ToggleSidebar), context.clone(), false, None, cx);
    let resolved = kitchen_shell::keys::resolve("canvas", CANVAS_KEYS, |key| load(key, cx).is_ok());
    let bindings: Vec<KeyBinding> = resolved.iter().flat_map(|(_, keys)| keys.iter().filter_map(|key| load(key, cx).ok())).collect();
    cx.bind_keys(bindings);
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
