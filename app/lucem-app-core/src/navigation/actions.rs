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

/// Every binding as `(group, &[(keys, what)])`, for the searchable Keyboard & mouse dialog.
pub const HELP: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("Ctrl+O", "Open an image"),
            ("Ctrl+Shift+O", "Open a project"),
            ("Ctrl+Shift+S", "Save the project in a folder"),
            ("Ctrl+K / F3", "Command palette: search any operation, tool, or action"),
            ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo any change"),
            ("Ctrl+1 / 2 / 3", "Switch workspace: Explore · Pipeline · Analyze"),
            ("Ctrl+R", "Toggle rulers on / off"),
            ("Ctrl+,", "Open Settings (theme, accent, memory, runtime)"),
            ("Ctrl+[ / Ctrl+]", "Show / hide the left (Layers) / right (studio) panels"),
            ("F1", "Open this Keyboard & mouse shortcuts search"),
        ],
    ),
    (
        "Navigation & View (on the image)",
        &[
            ("Wheel / Pinch", "Zoom at the pointer"),
            ("Space + Drag / Middle-drag", "Pan the image with any tool"),
            ("= / +  or  -", "Zoom in / zoom out"),
            ("F  or  Ctrl+0", "Fit image to view"),
            ("Ctrl+1", "Zoom to 100 % (actual pixels)"),
            ("PageUp / PageDown", "Previous / next Z-plane (3D volumes)"),
            ("A", "Toggle annotations overlay"),
            ("Shift+F", "Toggle annotation fill"),
            ("Shift+N", "Toggle annotation labels"),
            ("\\", "Peek Raw vs. Processed (toggle results overlay)"),
            ("Tab", "Hide / show both side panels"),
        ],
    ),
    (
        "Tools (one key each on the image)",
        &[
            ("V", "Select & move annotations (Shift adds)"),
            ("M", "Region select (Marquee box)"),
            ("H", "Hand (pan)"),
            ("Z", "Zoom (click in, Alt+click out)"),
            ("U", "Rectangle"),
            ("O", "Ellipse"),
            ("P", "Polygon (Enter / double-click closes, Backspace undoes corner)"),
            ("L", "Lasso (freehand closed region)"),
            ("N", "Points"),
            ("B", "Brush (Shift draws straight line)"),
            ("E", "Eraser (Shift erases straight line)"),
            ("W", "Wand (intensity-snapping brush)"),
            ("T", "Threshold (live segmentation)"),
            ("S", "Segment (SAM model)"),
            ("R", "Ruler (distance measurement)"),
            ("I", "Pixel probe (values & 3D cursor)"),
        ],
    ),
    (
        "Tool Sizing, Shapes & Numbers",
        &[
            ("[ / ]  or  Ctrl+- / Ctrl+=", "Decrease / increase active tool size (Brush, Eraser, Wand, Threshold smooth)"),
            ("Enter / Esc / Backspace", "Close polygon · cancel gesture · remove last polygon corner"),
            ("Delete", "Delete selected annotations"),
            ("Drag a number", "Scrub any numeric pill left/right (Shift = 5× coarse, Alt = 0.2× fine)"),
            ("Click a number", "Type an exact value directly (Enter commits, Esc cancels)"),
        ],
    ),
];
