//! Blender 4.x neutral dark charcoal theme (`palette`).
//!
//! Uses strictly neutral dark 1px recessed borders (`#111111` / `#1a1a1a`) and matte charcoal
//! surfaces (`#141414`, `#1d1d1d`, `#242424`, `#2b2b2b`, `#333333`) with Blender's `#4772b3`
//! selection fill — never colored/neon card edges.

use gpui_kit::{rgb, rgba, Hsla};

pub mod palette {
    use super::*;

    /// Deepest viewport void background (`#141414`).
    pub fn canvas_bg() -> Hsla {
        rgb(0x141414).into()
    }

    /// Subtle dark viewport grid (`#181818`).
    pub fn canvas_grid() -> Hsla {
        rgb(0x181818).into()
    }

    /// Primary editor/outliner panel surface (`#242424`).
    pub fn panel_bg() -> Hsla {
        rgb(0x242424).into()
    }

    /// Subpanel / collapsible region surface (`#2b2b2b`).
    pub fn surface_card() -> Hsla {
        rgb(0x2b2b2b).into()
    }

    /// Standard Blender button / widget fill (`#333333`).
    pub fn widget_bg() -> Hsla {
        rgb(0x333333).into()
    }

    /// Hovered button or subpanel surface (`#3d3d3d`).
    pub fn surface_hover() -> Hsla {
        rgb(0x3d3d3d).into()
    }

    /// Semi-transparent backdrop (`rgba(20, 20, 20, 0.90)`).
    pub fn glass_overlay() -> Hsla {
        rgba(0x141414e6).into()
    }

    /// Blender 1px recessed dark border (`#111111`).
    pub fn border_subtle() -> Hsla {
        rgb(0x111111).into()
    }

    /// Neutral subpanel divider (`#1a1a1a` — never colored).
    pub fn border_accent() -> Hsla {
        rgb(0x1a1a1a).into()
    }

    /// Blender 4.x active selection blue (`#4772b3`).
    pub fn accent_blue() -> Hsla {
        rgb(0x4772b3).into()
    }

    /// Neutral widget fill (`#333333`).
    pub fn accent_blue_soft() -> Hsla {
        rgb(0x333333).into()
    }

    /// Neutral indicator text (`#cccccc`).
    pub fn accent_emerald() -> Hsla {
        rgb(0xcccccc).into()
    }

    /// Neutral parameter badge text (`#c2c2c2`).
    pub fn accent_amber() -> Hsla {
        rgb(0xc2c2c2).into()
    }

    /// Primary crisp text (`#e5e5e5`).
    pub fn text_primary() -> Hsla {
        rgb(0xe5e5e5).into()
    }

    /// Secondary muted text (`#999999`).
    pub fn text_secondary() -> Hsla {
        rgb(0x999999).into()
    }

    /// Dimmed metadata / label text (`#6e6e6e`).
    pub fn text_dim() -> Hsla {
        rgb(0x6e6e6e).into()
    }

    /// Deepest canvas void background (`#141414`).
    pub fn canvas_void() -> Hsla {
        canvas_bg()
    }

    /// Alternate dark viewport surface (`#181818`).
    pub fn canvas_checker_a() -> Hsla {
        rgb(0x181818).into()
    }

    /// Top/bottom Blender header bar background (`#1d1d1d`).
    pub fn header_bg() -> Hsla {
        rgb(0x1d1d1d).into()
    }

    /// Subpanel background alias (`#2b2b2b`).
    pub fn card_bg() -> Hsla {
        surface_card()
    }

    /// Hovered widget background alias (`#3d3d3d`).
    pub fn card_hover_bg() -> Hsla {
        surface_hover()
    }

    /// Neutral dark divider (`#1a1a1a` — never colored).
    pub fn border_strong() -> Hsla {
        rgb(0x1a1a1a).into()
    }

    /// Semi-transparent modal scrim (`rgba(14, 14, 14, 0.85)`).
    pub fn modal_scrim() -> Hsla {
        rgba(0x0e0e0ed9).into()
    }

    /// Neutral bright label text (`#d8d8d8`).
    pub fn accent_cyan() -> Hsla {
        rgb(0xd8d8d8).into()
    }

    /// Neutral dark button fill (`#303030`).
    pub fn accent_cyan_muted() -> Hsla {
        rgb(0x303030).into()
    }

    /// Muted close/remove button red (`#b33a3a`).
    pub fn accent_coral() -> Hsla {
        rgb(0xb33a3a).into()
    }
}
