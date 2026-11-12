//! Device kernels for marching-squares and marching-cubes active cell classification.

use cubecl::prelude::*;
use lucem_core::compute::element_position;

/// Computes the 4-bit marching-squares cell mask (`0..16`) for each `(r, c)` cell in a `[h - 1, w - 1]` grid.
#[cube(launch)]
pub fn marching_squares_cell_mask_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    cell_mask: &mut [u32],
    w: u32,
    cell_w: u32,
    level: F,
    num_cells: u32,
) {
    let i = element_position();
    if (i as u32) < num_cells {
        let idx = i as u32;
        let r = idx / cell_w;
        let c = idx % cell_w;
        let v00 = input[(r * w + c) as usize] >= level;
        let v01 = input[(r * w + c + 1u32) as usize] >= level;
        let v11 = input[((r + 1u32) * w + c + 1u32) as usize] >= level;
        let v10 = input[((r + 1u32) * w + c) as usize] >= level;
        let mut mask = 0u32;
        if v00 {
            mask |= 1u32;
        }
        if v01 {
            mask |= 2u32;
        }
        if v11 {
            mask |= 4u32;
        }
        if v10 {
            mask |= 8u32;
        }
        cell_mask[i] = mask;
    }
}

/// Computes the 8-bit marching-cubes voxel cell mask (`0..256`) for each `(z, y, x)` cell in a
/// `[d - 1, h - 1, w - 1]` grid.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn marching_cubes_cell_mask_kernel<F: lucem_core::element::ImageFloat>(
    input: &[F],
    cell_mask: &mut [u32],
    h: u32,
    w: u32,
    cell_h: u32,
    cell_w: u32,
    level: F,
    num_cells: u32,
) {
    let i = element_position();
    if (i as u32) < num_cells {
        let idx = i as u32;
        let plane = cell_h * cell_w;
        let z = idx / plane;
        let rem = idx % plane;
        let y = rem / cell_w;
        let x = rem % cell_w;
        let slice = h * w;

        let c0 = input[(z * slice + y * w + x) as usize] >= level;
        let c1 = input[(z * slice + y * w + x + 1u32) as usize] >= level;
        let c2 = input[(z * slice + (y + 1u32) * w + x + 1u32) as usize] >= level;
        let c3 = input[(z * slice + (y + 1u32) * w + x) as usize] >= level;
        let c4 = input[((z + 1u32) * slice + y * w + x) as usize] >= level;
        let c5 = input[((z + 1u32) * slice + y * w + x + 1u32) as usize] >= level;
        let c6 = input[((z + 1u32) * slice + (y + 1u32) * w + x + 1u32) as usize] >= level;
        let c7 = input[((z + 1u32) * slice + (y + 1u32) * w + x) as usize] >= level;

        let mut mask = 0u32;
        if c0 {
            mask |= 1u32;
        }
        if c1 {
            mask |= 2u32;
        }
        if c2 {
            mask |= 4u32;
        }
        if c3 {
            mask |= 8u32;
        }
        if c4 {
            mask |= 16u32;
        }
        if c5 {
            mask |= 32u32;
        }
        if c6 {
            mask |= 64u32;
        }
        if c7 {
            mask |= 128u32;
        }
        cell_mask[i] = mask;
    }
}
