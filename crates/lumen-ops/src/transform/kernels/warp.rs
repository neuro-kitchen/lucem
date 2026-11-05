//! Coordinate generation kernels for 2-D geometric warps (affine/projective, polar/log-polar, swirl).

use cubecl::prelude::*;
use lumen_core::compute::element_position;

/// Generates inverse-mapped 2-D source coordinates `(coords_y[i], coords_x[i])` for an output grid
/// `[out_h, out_w]` under a $3 \times 3$ homogeneous inverse matrix `m[0..9]` (row-major `[y, x, 1]`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn affine_coords_2d_kernel<F: Float>(
    m: &[F],
    coords_y: &mut [F],
    coords_x: &mut [F],
    out_w: u32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let oy = F::cast_from(idx / out_w);
        let ox = F::cast_from(idx % out_w);
        let num_y = m[0] * oy + m[1] * ox + m[2];
        let num_x = m[3] * oy + m[4] * ox + m[5];
        let denom = m[6] * oy + m[7] * ox + m[8];
        let safe_denom = if F::abs(denom) <= F::new(1e-12f32) {
            F::new(1.0f32)
        } else {
            denom
        };
        coords_y[i] = num_y / safe_denom;
        coords_x[i] = num_x / safe_denom;
    }
}

/// Generates source `(y, x)` coordinates for a polar (`log_polar = false`) or log-polar
/// (`log_polar = true`) image of shape `[n_angles, n_radii]` centered at `(cy, cx)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn polar_coords_2d_kernel<F: Float>(
    coords_y: &mut [F],
    coords_x: &mut [F],
    cy: F,
    cx: F,
    max_radius: F,
    n_angles: u32,
    n_radii: u32,
    total: u32,
    #[comptime] log_polar: bool,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let row = idx / n_radii;
        let col = idx % n_radii;
        let two_pi = 6.283_185_5f32;
        let angle = (row as f32) * two_pi / (u32::max(n_angles, 1u32) as f32);
        let frac = (col as f32) / (u32::max(n_radii - 1u32, 1u32) as f32);
        let max_r = f32::cast_from(max_radius);
        let r = if comptime!(log_polar) {
            f32::exp(frac * f32::ln(f32::max(max_r, 1.0f32) + 1.0f32)) - 1.0f32
        } else {
            frac * max_r
        };
        let dy = F::cast_from(r * f32::sin(angle));
        let dx = F::cast_from(r * f32::cos(angle));
        coords_y[i] = cy + dy;
        coords_x[i] = cx + dx;
    }
}

/// Generates inverse-mapped source `(y, x)` coordinates for a non-linear swirl warp around `(cy, cx)`:
/// $\theta'(r) = \theta + \text{rotation} + \text{strength} \cdot \exp(-r / \text{radius_decay})$.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn swirl_coords_2d_kernel<F: Float>(
    coords_y: &mut [F],
    coords_x: &mut [F],
    cy: F,
    cx: F,
    strength: F,
    radius_decay: F,
    rotation: F,
    width: u32,
    total: u32,
) {
    let i = element_position();
    let idx = i as u32;
    if idx < total {
        let oy = F::cast_from(idx / width);
        let ox = F::cast_from(idx % width);
        let dy = f32::cast_from(oy - cy);
        let dx = f32::cast_from(ox - cx);
        let r = f32::sqrt(dy * dy + dx * dx);
        let decay = f32::max(f32::cast_from(radius_decay), 1e-6f32);
        let delta_theta = f32::cast_from(rotation) + f32::cast_from(strength) * f32::exp(-r / decay);
        let c = f32::cos(delta_theta);
        let s = f32::sin(delta_theta);
        let src_dx = c * dx - s * dy;
        let src_dy = s * dx + c * dy;
        coords_y[i] = cy + F::cast_from(src_dy);
        coords_x[i] = cx + F::cast_from(src_dx);
    }
}
