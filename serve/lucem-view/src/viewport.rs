//! Interactive 2-D viewport camera: cursor-anchored zoom in/out, pan, fit-to-canvas, `1:1` pixel
//! zoom, and bidirectional `screen <-> image` coordinate transforms.

use serde::{Deserialize, Serialize};

/// Camera state for the interactive Studio / Explore canvas.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewportCamera {
    /// Zoom scale factor (`1.0` = 100% 1:1 native pixels; `0.05..=64.0`).
    pub zoom: f32,
    /// Horizontal pan offset in screen pixels relative to the centered canvas origin.
    pub pan_x: f32,
    /// Vertical pan offset in screen pixels relative to the centered canvas origin.
    pub pan_y: f32,
    /// Whether to display the dark checkerboard/grid background under the image.
    pub show_grid: bool,
    /// Whether to overlay Marching Squares vector iso-contours on top of the image.
    pub show_contours: bool,
    /// Whether to overlay connected-component bounding boxes & centroids on top of the image.
    pub show_region_boxes: bool,
}

impl Default for ViewportCamera {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            show_grid: true,
            show_contours: false,
            show_region_boxes: false,
        }
    }
}

impl ViewportCamera {
    pub const MIN_ZOOM: f32 = 0.05;
    pub const MAX_ZOOM: f32 = 64.0;

    /// Clamps `zoom` to `[MIN_ZOOM, MAX_ZOOM]`.
    pub fn set_zoom(&mut self, new_zoom: f32) {
        self.zoom = new_zoom.clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
    }

    /// Zooms by `factor` (e.g. `1.2` for zoom in, `1.0 / 1.2` for zoom out) anchored around the
    /// cursor offset `(cursor_rel_x, cursor_rel_y)` relative to the viewport center so the image
    /// point under the cursor stays stationary.
    pub fn zoom_at_cursor_offset(&mut self, factor: f32, cursor_rel_x: f32, cursor_rel_y: f32) {
        let old_zoom = self.zoom;
        let new_zoom = (old_zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let ratio = new_zoom / old_zoom;
        self.pan_x = cursor_rel_x - (cursor_rel_x - self.pan_x) * ratio;
        self.pan_y = cursor_rel_y - (cursor_rel_y - self.pan_y) * ratio;
        self.zoom = new_zoom;
    }

    /// Zooms in or out around the center of the viewport.
    pub fn zoom_step(&mut self, factor: f32) {
        self.zoom_at_cursor_offset(factor, 0.0, 0.0);
    }

    /// Resets pan to `(0, 0)` and sets zoom to `1.0` (100% 1:1 pixel accuracy).
    pub fn reset_one_to_one(&mut self) {
        self.zoom = 1.0;
        self.pan_x = 0.0;
        self.pan_y = 0.0;
    }

    /// Fits an image of `(img_w, img_h)` inside a viewport of `(view_w, view_h)` with `margin_px`
    /// padding on each side.
    pub fn fit_to_viewport(&mut self, img_w: usize, img_h: usize, view_w: f32, view_h: f32, margin_px: f32) {
        let usable_w = (view_w - 2.0 * margin_px).max(64.0);
        let usable_h = (view_h - 2.0 * margin_px).max(64.0);
        let sx = usable_w / (img_w.max(1) as f32);
        let sy = usable_h / (img_h.max(1) as f32);
        self.zoom = sx.min(sy).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        self.pan_x = 0.0;
        self.pan_y = 0.0;
    }

    /// Translates the camera by `(dx, dy)` screen pixels.
    pub fn pan_by(&mut self, dx: f32, dy: f32) {
        self.pan_x += dx;
        self.pan_y += dy;
    }

    /// Computes the `(left, top, scaled_width, scaled_height)` rectangle in viewport-local pixels
    /// for an image of `(img_w, img_h)` inside a viewport of size `(view_w, view_h)`.
    pub fn image_rect(&self, img_w: usize, img_h: usize, view_w: f32, view_h: f32) -> (f32, f32, f32, f32) {
        let sw = (img_w as f32) * self.zoom;
        let sh = (img_h as f32) * self.zoom;
        let left = 0.5 * (view_w - sw) + self.pan_x;
        let top = 0.5 * (view_h - sh) + self.pan_y;
        (left, top, sw, sh)
    }

    /// Maps a viewport-local point `(local_x, local_y)` to fractional image coordinates `(img_y, img_x)`
    /// inside an image of `(img_w, img_h)` rendered in a viewport of `(view_w, view_h)`.
    pub fn screen_to_image(
        &self,
        local_x: f32,
        local_y: f32,
        img_w: usize,
        img_h: usize,
        view_w: f32,
        view_h: f32,
    ) -> Option<(usize, usize)> {
        let (left, top, sw, sh) = self.image_rect(img_w, img_h, view_w, view_h);
        if sw <= 0.0 || sh <= 0.0 {
            return None;
        }
        let rx = (local_x - left) / self.zoom;
        let ry = (local_y - top) / self.zoom;
        if rx >= 0.0 && rx < (img_w as f32) && ry >= 0.0 && ry < (img_h as f32) {
            Some((ry.floor() as usize, rx.floor() as usize))
        } else {
            None
        }
    }

    /// Formats the zoom percentage for UI badges (e.g. `"100%"`, `"250%"`).
    pub fn zoom_percent_label(&self) -> String {
        format!("{:.0}%", self.zoom * 100.0)
    }
}
