//! Dark pro-studio color palette (ArtCraft × VectorCraft × Blender obsidian theme).

use gpui_kit::{rgb, rgba, Hsla};

pub mod palette {
    use super::*;

    /// Deepest canvas void background (`#0b0d12`).
    pub fn canvas_bg() -> Hsla {
        rgb(0x0b0d12).into()
    }

    /// Subtle checkerboard/grid line on the blank canvas (`#151922`).
    pub fn canvas_grid() -> Hsla {
        rgb(0x151922).into()
    }

    /// Primary panel/dock surface (`#12151e`).
    pub fn panel_bg() -> Hsla {
        rgb(0x12151e).into()
    }

    /// Elevated card or floating modal surface (`#181c28`).
    pub fn surface_card() -> Hsla {
        rgb(0x181c28).into()
    }

    /// Hovered or active card surface (`#202636`).
    pub fn surface_hover() -> Hsla {
        rgb(0x202636).into()
    }

    /// Semi-transparent glass backdrop (`rgba(14, 17, 25, 0.86)`).
    pub fn glass_overlay() -> Hsla {
        rgba(0x0e1119dc).into()
    }

    /// Hairline border color (`#262c3d`).
    pub fn border_subtle() -> Hsla {
        rgb(0x262c3d).into()
    }

    /// Highlighted border color (`#3b82f6`).
    pub fn border_accent() -> Hsla {
        rgb(0x3b82f6).into()
    }

    /// Primary electric blue accent (`#3b82f6`).
    pub fn accent_blue() -> Hsla {
        rgb(0x3b82f6).into()
    }

    /// Soft accent fill (`rgba(59, 130, 246, 0.18)`).
    pub fn accent_blue_soft() -> Hsla {
        rgba(0x3b82f62e).into()
    }

    /// Emerald/cyan scientific accent (`#10b981`).
    pub fn accent_emerald() -> Hsla {
        rgb(0x10b981).into()
    }

    /// Amber Blender modifier accent (`#f59e0b`).
    pub fn accent_amber() -> Hsla {
        rgb(0xf59e0b).into()
    }

    /// Primary crisp text (`#f1f5f9`).
    pub fn text_primary() -> Hsla {
        rgb(0xf1f5f9).into()
    }

    /// Secondary muted text (`#94a3b8`).
    pub fn text_secondary() -> Hsla {
        rgb(0x94a3b8).into()
    }

    /// Dimmed metadata / ruler text (`#64748b`).
    pub fn text_dim() -> Hsla {
        rgb(0x64748b).into()
    }

    /// Deepest canvas void background (`#0b0d12`).
    pub fn canvas_void() -> Hsla {
        canvas_bg()
    }

    /// Alternate dark checkerboard cell (`#111520`).
    pub fn canvas_checker_a() -> Hsla {
        rgb(0x111520).into()
    }

    /// Top/bottom header bar background (`#0e1118`).
    pub fn header_bg() -> Hsla {
        rgb(0x0e1118).into()
    }

    /// Card background alias (`#181c28`).
    pub fn card_bg() -> Hsla {
        surface_card()
    }

    /// Hovered card background alias (`#202636`).
    pub fn card_hover_bg() -> Hsla {
        surface_hover()
    }

    /// Stronger border color (`#334155`).
    pub fn border_strong() -> Hsla {
        rgb(0x334155).into()
    }

    /// Semi-transparent modal scrim (`rgba(11, 13, 18, 0.84)`).
    pub fn modal_scrim() -> Hsla {
        rgba(0x0b0d12d6).into()
    }

    /// Bright cyan scientific accent (`#22d3ee`).
    pub fn accent_cyan() -> Hsla {
        rgb(0x22d3ee).into()
    }

    /// Muted cyan fill (`rgba(34, 211, 238, 0.16)`).
    pub fn accent_cyan_muted() -> Hsla {
        rgba(0x22d3ee29).into()
    }

    /// Coral destructive/reset accent (`#f43f5e`).
    pub fn accent_coral() -> Hsla {
        rgb(0xf43f5e).into()
    }
}
