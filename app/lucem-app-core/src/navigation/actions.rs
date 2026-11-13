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

/// Binds the canvas' own keys.
pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([KeyBinding::new("tab", ToggleSidebar, Some("ImageCanvas"))]);
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
