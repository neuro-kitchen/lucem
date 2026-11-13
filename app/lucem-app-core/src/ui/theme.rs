//! lucem's look: the neuro-kitchen design tokens from `kitchen_ui::theme` (spacing, type, radii,
//! plain-grey neutrals, the accent palette and its WCAG checks), with lucem's blue accent and the
//! sizes and pasteboard only lucem has.

use std::sync::atomic::{AtomicU8, Ordering};

use gpui_kit::{App, Window};
pub use kitchen_ui::theme::{contrast, is_dark, rgb8, rgb_hsla, sp, system_dark, Look, Palette, Rgb, FONT_RANGE, FONT_SIZE, LG, MD, PANE_GAP, PANE_PADDING, RADIUS, RADIUS_LG, RADIUS_SM, SM, TARGET_MIN, XL, XS, XXL};

/// Heights of the fixed bars (px).
pub const TITLE_BAR_H: f32 = 36.;
pub const OPTIONS_BAR_H: f32 = 36.;
pub const STATUS_BAR_H: f32 = 30.;
/// Tool bar width and its buttons (px): 32 px targets.
pub const TOOLBAR_W: f32 = 44.;
pub const TOOL_BUTTON: f32 = 32.;
/// Rulers (px).
pub const RULER: f32 = 20.;
/// Right sidebar width (px).
pub const SIDEBAR_W: f32 = 304.;

/// Pasteboards offered in Settings (what is around the image).
pub const PASTEBOARDS: [&str; 3] = ["Grey", "Dark", "Light"];

/// lucem's accent: blue (the user can pick another in Settings). Made with
/// [`Palette::from_rgb`], so it passes contrast in both themes.
pub fn palette() -> Palette {
    Palette::from_rgb("Blue", 0x3d65a8)
}

static PASTEBOARD: AtomicU8 = AtomicU8::new(0);

/// Sets the look from the preferences: dark or light (the system's appearance when the theme
/// follows it), the user's accent, the text size and the pasteboard. [`apply`] then hands it to
/// the component library.
pub fn configure(prefs: &crate::data::prefs::Prefs, window: &Window) {
    let dark = prefs.theme.unwrap_or_else(|| system_dark(window));
    kitchen_ui::theme::set_look(Look::new(palette()).with_dark(dark).with_accent(prefs.accent).with_font_size(prefs.font_size));
    PASTEBOARD.store(prefs.pasteboard, Ordering::Relaxed);
}

/// Applies the look ([`configure`]) to the component library and redraws.
pub fn apply(window: &mut Window, cx: &mut App) {
    kitchen_ui::theme::apply(cx);
    window.refresh();
}

/// The colours: kitchen-ui's neutrals and accent, and lucem's pasteboard.
pub mod color {
    pub use kitchen_ui::theme::color::*;
    use gpui_kit::Hsla;

    /// The pasteboard behind the image (Affinity style): grey by default, so an image with black
    /// edges reads as a page on it.
    pub fn canvas() -> Hsla {
        let pick = |dark: u32, light: u32| rgb_hsla(if super::is_dark() { dark } else { light });
        match super::PASTEBOARD.load(std::sync::atomic::Ordering::Relaxed) {
            1 => pick(0x161616, 0x2a2a2a),
            2 => pick(0x8a8a8a, 0xf0f0f0),
            _ => pick(0x3a3a3a, 0xbcbcbc),
        }
    }
}
