//! Design tokens (`.tasks/2026-11-06_ui-ux-redesign.md` §5) and their application to the
//! component library's theme, so every gpui-kit component (buttons, menus, inputs, tooltips)
//! and our own layout use one set of values.
//!
//! - Spacing on a 4 px base: [`XS`] 4, [`SM`] 8, [`MD`] 12, [`LG`] 16, [`XL`] 24.
//! - Type: base 15 px, so `text_xs` ≈ 11 (captions), `text_sm` ≈ 13 (body), `text_base` 15
//!   (titles), `text_lg` ≈ 17 (headings); weights 400 and 600 only.
//! - Surfaces step clearly apart; borders are lighter than surfaces; secondary text passes WCAG
//!   AA (≥ 4.5:1) on every surface; one accent.

use gpui_kit::component::{Theme, ThemeMode};
use gpui_kit::{px, rgb, rgba, App, Hsla, Pixels, Window};

/// Spacing scale (px).
pub const XS: f32 = 4.;
pub const SM: f32 = 8.;
pub const MD: f32 = 12.;
pub const LG: f32 = 16.;
pub const XL: f32 = 24.;

/// Heights of the fixed bars (px).
pub const TITLE_BAR_H: f32 = 36.;
pub const OPTIONS_BAR_H: f32 = 36.;
pub const STATUS_BAR_H: f32 = 26.;
/// Tool bar width and its buttons (px): 32 px targets.
pub const TOOLBAR_W: f32 = 44.;
pub const TOOL_BUTTON: f32 = 32.;
/// Rulers (px).
pub const RULER: f32 = 20.;
/// Right sidebar width (px).
pub const SIDEBAR_W: f32 = 304.;

/// What the user chose in Settings, read by every colour (switched at run time).
mod look {
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};

    pub static DARK: AtomicBool = AtomicBool::new(true);
    pub static ACCENT: AtomicU32 = AtomicU32::new(0x3d65a8);
    pub static PASTEBOARD: AtomicU8 = AtomicU8::new(0);
    pub static FONT: AtomicU32 = AtomicU32::new(15);

    pub fn dark() -> bool {
        DARK.load(Ordering::Relaxed)
    }
}

/// Accent colours offered in Settings: name and RGB.
pub const ACCENTS: [(&str, u32); 6] = [("Blue", 0x3d65a8), ("Teal", 0x2b7f86), ("Green", 0x3f8a52), ("Purple", 0x7451b8), ("Orange", 0xb8622a), ("Pink", 0xb24470)];
/// Pasteboards offered in Settings (what is around the image).
pub const PASTEBOARDS: [&str; 3] = ["Grey", "Dark", "Light"];

/// Sets the look: dark or light, the accent (RGB), the pasteboard (index of [`PASTEBOARDS`]) and
/// the interface text size (px). [`apply`] then hands it to the component library.
pub fn set_look(dark: bool, accent: u32, pasteboard: u8, font: f32) {
    use std::sync::atomic::Ordering;
    look::DARK.store(dark, Ordering::Relaxed);
    look::ACCENT.store(accent, Ordering::Relaxed);
    look::PASTEBOARD.store(pasteboard, Ordering::Relaxed);
    look::FONT.store(font.round().clamp(12.0, 18.0) as u32, Ordering::Relaxed);
}

/// Sets the look from the preferences (the system's appearance when the theme follows it).
pub fn configure(prefs: &crate::data::prefs::Prefs, window: &Window) {
    use gpui_kit::WindowAppearance as W;
    let dark = prefs.theme.unwrap_or_else(|| matches!(window.appearance(), W::Dark | W::VibrantDark));
    set_look(dark, prefs.accent, prefs.pasteboard, prefs.font_size);
}

pub fn is_dark() -> bool {
    look::dark()
}

/// The surfaces, from the deepest: dark values, then light ones.
pub mod color {
    use super::*;

    fn pick(dark: u32, light: u32) -> Hsla {
        rgb(if look::dark() { dark } else { light }).into()
    }

    /// The pasteboard behind the image (Affinity style): grey by default, so an image with black
    /// edges reads as a page on it.
    pub fn canvas() -> Hsla {
        match look::PASTEBOARD.load(std::sync::atomic::Ordering::Relaxed) {
            1 => pick(0x161616, 0x2a2a2a),
            2 => pick(0x8a8a8a, 0xf0f0f0),
            _ => pick(0x3a3a3a, 0xbcbcbc),
        }
    }
    /// Title bar, tool column, rulers, tab strips.
    pub fn chrome() -> Hsla {
        pick(0x222222, 0xe4e4e4)
    }
    /// Panels.
    pub fn panel() -> Hsla {
        pick(0x2a2a2a, 0xf2f2f2)
    }
    /// Raised blocks inside panels, popovers, panel group headers.
    pub fn raised() -> Hsla {
        pick(0x313131, 0xfafafa)
    }
    /// Controls (buttons, inputs).
    pub fn control() -> Hsla {
        pick(0x3a3a3a, 0xe0e0e0)
    }
    pub fn hover() -> Hsla {
        pick(0x454545, 0xd4d4d4)
    }
    /// Separators: darker than the surfaces they divide in dark mode (Affinity's thin lines).
    pub fn border() -> Hsla {
        pick(0x171717, 0xc6c6c6)
    }
    pub fn text() -> Hsla {
        pick(0xe6e6e6, 0x1c1c1c)
    }
    /// Secondary text: ≥ 5:1 on controls and panels in both modes.
    pub fn text_muted() -> Hsla {
        pick(0xb0b0b0, 0x555555)
    }
    /// Disabled only.
    pub fn text_disabled() -> Hsla {
        pick(0x6e6e6e, 0x9a9a9a)
    }
    /// The one accent: selection, the active tool, primary actions (white text on it).
    pub fn accent() -> Hsla {
        rgb(look::ACCENT.load(std::sync::atomic::Ordering::Relaxed)).into()
    }
    pub fn accent_hover() -> Hsla {
        let a = accent();
        Hsla { l: (a.l + 0.07).min(0.9), ..a }
    }
    pub fn danger() -> Hsla {
        rgb(0xd65c5c).into()
    }
    pub fn success() -> Hsla {
        rgb(0x4caf7a).into()
    }
    pub fn warning() -> Hsla {
        rgb(0xe0a030).into()
    }
}

/// Base font size (px) by default.
pub const FONT_SIZE: f32 = 15.;

/// Applies the look ([`set_look`]) and the tokens to the component library, and redraws.
pub fn apply(window: &mut Window, cx: &mut App) {
    Theme::change(if is_dark() { ThemeMode::Dark } else { ThemeMode::Light }, Some(window), cx);
    let font = look::FONT.load(std::sync::atomic::Ordering::Relaxed) as f32;
    Theme::update(cx, |t| {
        t.font_size = px(font);
        t.radius = px(5.);
        t.radius_lg = px(8.);
        t.shadow = true;
        let c = &mut t.colors;
        c.background = color::panel();
        c.foreground = color::text();
        c.muted = color::raised();
        c.muted_foreground = color::text_muted();
        c.border = color::border();
        c.input = color::hover();
        c.ring = color::accent();
        c.caret = color::text();
        c.selection = color::accent().opacity(0.45);
        c.popover = color::raised();
        c.popover_foreground = color::text();
        c.sidebar = color::chrome();
        c.sidebar_foreground = color::text();
        c.sidebar_border = color::border();
        c.sidebar_accent = color::control();
        c.sidebar_accent_foreground = color::text();
        c.sidebar_primary = color::accent();
        c.sidebar_primary_foreground = rgb(0xffffff).into();
        c.primary = color::accent();
        c.primary_hover = color::accent_hover();
        c.primary_active = color::accent();
        c.primary_foreground = rgb(0xffffff).into();
        c.secondary = color::control();
        c.secondary_hover = color::hover();
        c.secondary_active = color::hover();
        c.secondary_foreground = color::text();
        c.accent = color::control();
        c.accent_foreground = color::text();
        c.title_bar = color::chrome();
        c.title_bar_border = color::border();
        c.tab_bar = color::chrome();
        c.tab = color::chrome();
        c.tab_active = color::panel();
        c.tab_foreground = color::text_muted();
        c.tab_active_foreground = color::text();
        c.list = color::panel();
        c.list_even = color::panel();
        c.list_hover = color::control();
        c.list_active = color::accent().opacity(0.25);
        c.list_active_border = color::accent();
        c.list_head = color::raised();
        c.table = color::panel();
        c.table_even = color::raised();
        c.table_head = color::raised();
        c.table_head_foreground = color::text_muted();
        c.table_active = color::accent().opacity(0.25);
        c.table_active_border = color::accent();
        c.danger = color::danger();
        c.danger_foreground = rgb(0xffffff).into();
        c.success = color::success();
        c.success_foreground = rgb(0xffffff).into();
        c.warning = color::warning();
        c.info = color::accent();
        c.link = rgb(0x7fa6e6).into();
        c.scrollbar = rgba(0x00000000).into();
        c.scrollbar_thumb = if is_dark() { rgba(0xffffff33).into() } else { rgba(0x00000033).into() };
        c.scrollbar_thumb_hover = if is_dark() { rgba(0xffffff55).into() } else { rgba(0x00000055).into() };
        c.window_border = color::border();
        c.overlay = rgba(0x00000080).into();
        c.drag_border = color::accent();
        c.drop_target = color::accent().opacity(0.2);
        c.chart_grid = color::border();
    });
    window.refresh();
}

/// `px` of a token.
pub fn sp(v: f32) -> Pixels {
    px(v)
}

/// A colour as `Hsla` from 0–255 RGB.
pub fn rgb8(c: [u8; 3]) -> Hsla {
    rgb(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32).into()
}
