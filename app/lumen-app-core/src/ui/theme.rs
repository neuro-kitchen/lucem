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

/// The surfaces, from the deepest.
pub mod color {
    use super::*;

    /// The image canvas behind the image.
    pub fn canvas() -> Hsla {
        rgb(0x121212).into()
    }
    /// Title bar, tool bar, rulers.
    pub fn chrome() -> Hsla {
        rgb(0x181818).into()
    }
    /// Panels.
    pub fn panel() -> Hsla {
        rgb(0x1f1f1f).into()
    }
    /// Raised blocks inside panels, popovers.
    pub fn raised() -> Hsla {
        rgb(0x262626).into()
    }
    /// Controls (buttons, inputs).
    pub fn control() -> Hsla {
        rgb(0x2e2e2e).into()
    }
    pub fn hover() -> Hsla {
        rgb(0x383838).into()
    }
    pub fn border() -> Hsla {
        rgb(0x3a3a3a).into()
    }
    pub fn text() -> Hsla {
        rgb(0xe6e6e6).into()
    }
    /// Secondary text: 4.6:1 on controls, 6.7:1 on panels.
    pub fn text_muted() -> Hsla {
        rgb(0xa0a0a0).into()
    }
    /// Disabled only.
    pub fn text_disabled() -> Hsla {
        rgb(0x6e6e6e).into()
    }
    /// The one accent: selection, the active tool, primary actions (white text 5.8:1).
    pub fn accent() -> Hsla {
        rgb(0x3d65a8).into()
    }
    pub fn accent_hover() -> Hsla {
        rgb(0x4a72b5).into()
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

/// Base font size (rem).
pub const FONT_SIZE: f32 = 15.;

/// Applies the dark theme and the tokens to the component library.
pub fn apply(window: &mut Window, cx: &mut App) {
    Theme::change(ThemeMode::Dark, Some(window), cx);
    Theme::update(cx, |t| {
        t.font_size = px(FONT_SIZE);
        t.radius = px(5.);
        t.radius_lg = px(8.);
        t.shadow = true;
        let c = &mut t.colors;
        c.background = color::panel();
        c.foreground = color::text();
        c.muted = color::raised();
        c.muted_foreground = color::text_muted();
        c.border = color::border();
        c.input = color::border();
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
        c.primary_active = rgb(0x355a96).into();
        c.primary_foreground = rgb(0xffffff).into();
        c.secondary = color::control();
        c.secondary_hover = color::hover();
        c.secondary_active = rgb(0x333333).into();
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
        c.table_even = rgb(0x222222).into();
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
        c.scrollbar_thumb = rgba(0xffffff33).into();
        c.scrollbar_thumb_hover = rgba(0xffffff55).into();
        c.window_border = color::border();
        c.overlay = rgba(0x00000080).into();
        c.drag_border = color::accent();
        c.drop_target = color::accent().opacity(0.2);
        c.chart_grid = color::border();
    });
}

/// `px` of a token.
pub fn sp(v: f32) -> Pixels {
    px(v)
}

/// A colour as `Hsla` from 0–255 RGB.
pub fn rgb8(c: [u8; 3]) -> Hsla {
    rgb(((c[0] as u32) << 16) | ((c[1] as u32) << 8) | c[2] as u32).into()
}
