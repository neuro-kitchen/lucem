//! Isocontour extraction in 2-D (`marching_squares`) and isosurface mesh extraction in 3-D
//! (`marching_cubes`, `mesh_surface_area`).

use std::collections::HashMap;

use cubecl::prelude::*;
use lumen_core::compute::LaunchGeometry;
use lumen_core::element::{cast, to_f64, ImageFloat};
use lumen_core::memory::DeviceArray;
use lumen_core::Shape;

use super::kernels::{marching_cubes_cell_mask_kernel, marching_squares_cell_mask_kernel};
use crate::core::check;
use crate::error::{OpsError, OpsResult};

/// A 2-D polygonal contour extracted by [`marching_squares`]: an ordered sequence of `(y, x)`
/// subpixel vertices.
pub type Contour2d = Vec<[f64; 2]>;

/// Triangle mesh extracted by [`marching_cubes`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMesh {
    /// Vertex coordinates `[z, y, x]` in physical/voxel space.
    pub vertices: Vec<[f64; 3]>,
    /// Triangle index triples `[i0, i1, i2]` into `vertices`.
    pub faces: Vec<[usize; 3]>,
    /// Unit outward vertex normals `[nz, ny, nx]`.
    pub normals: Vec<[f64; 3]>,
}

/// Extracts 2-D iso-valued contours at `level` from a `[h, w]` image using the Marching Squares
/// algorithm. Active cells are classified on the device before assembling connected polylines.
pub fn marching_squares<F: ImageFloat>(
    client: &Client,
    input: &DeviceArray<F>,
    level: f64,
) -> OpsResult<Vec<Contour2d>> {
    check::ndim_in("marching_squares", input.shape(), &[2], "2")?;
    let (h, w) = (input.dims()[0], input.dims()[1]);
    if h < 2 || w < 2 {
        return Ok(Vec::new());
    }
    if !level.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "level",
            reason: "contour level must be finite".to_string(),
        });
    }

    let cell_h = h - 1;
    let cell_w = w - 1;
    let num_cells = cell_h * cell_w;
    let mask_dev = DeviceArray::<u32>::empty(client, Shape::new([cell_h, cell_w]))?;
    let geom = LaunchGeometry::elementwise(client, num_cells);
    unsafe {
        marching_squares_cell_mask_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            input.binding(),
            mask_dev.binding(),
            w as u32,
            cell_w as u32,
            cast::<F>(level),
            num_cells as u32,
        );
    }

    let masks = mask_dev.to_host(client);
    let vals: Vec<f64> = input.to_host(client).into_iter().map(to_f64).collect();

    let interp = |v0: f64, v1: f64| -> f64 {
        let denom = v1 - v0;
        if denom.abs() <= 1e-12 {
            0.5
        } else {
            ((level - v0) / denom).clamp(0.0, 1.0)
        }
    };

    // Edge endpoints in cell (r, c):
    // 0: top    (r, c) -> (r, c+1)
    // 1: right  (r, c+1) -> (r+1, c+1)
    // 2: bottom (r+1, c) -> (r+1, c+1)
    // 3: left   (r, c) -> (r+1, c)
    let mut segments: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for r in 0..cell_h {
        for c in 0..cell_w {
            let code = masks[r * cell_w + c];
            if code == 0 || code == 15 {
                continue;
            }
            let v00 = vals[r * w + c];
            let v01 = vals[r * w + c + 1];
            let v11 = vals[(r + 1) * w + c + 1];
            let v10 = vals[(r + 1) * w + c];

            let rf = r as f64;
            let cf = c as f64;
            let e_top = [rf, cf + interp(v00, v01)];
            let e_right = [rf + interp(v01, v11), cf + 1.0];
            let e_bottom = [rf + 1.0, cf + interp(v10, v11)];
            let e_left = [rf + interp(v00, v10), cf];

            let center = 0.25 * (v00 + v01 + v11 + v10);
            match code {
                1 | 14 => segments.push((e_left, e_top)),
                2 | 13 => segments.push((e_top, e_right)),
                4 | 11 => segments.push((e_right, e_bottom)),
                8 | 7 => segments.push((e_bottom, e_left)),
                3 | 12 => segments.push((e_left, e_right)),
                6 | 9 => segments.push((e_top, e_bottom)),
                5 => {
                    if center >= level {
                        segments.push((e_left, e_bottom));
                        segments.push((e_top, e_right));
                    } else {
                        segments.push((e_left, e_top));
                        segments.push((e_right, e_bottom));
                    }
                }
                10 => {
                    if center >= level {
                        segments.push((e_left, e_top));
                        segments.push((e_right, e_bottom));
                    } else {
                        segments.push((e_left, e_bottom));
                        segments.push((e_top, e_right));
                    }
                }
                _ => {}
            }
        }
    }

    // Stitch segments into connected polylines
    let mut used = vec![false; segments.len()];
    let mut contours = Vec::new();
    let close_pt = |a: [f64; 2], b: [f64; 2]| -> bool {
        (a[0] - b[0]).abs() <= 1e-6 && (a[1] - b[1]).abs() <= 1e-6
    };

    for i in 0..segments.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut chain = vec![segments[i].0, segments[i].1];
        loop {
            let mut extended = false;
            let tail = *chain.last().unwrap();
            for j in 0..segments.len() {
                if used[j] {
                    continue;
                }
                if close_pt(tail, segments[j].0) {
                    used[j] = true;
                    chain.push(segments[j].1);
                    extended = true;
                    break;
                } else if close_pt(tail, segments[j].1) {
                    used[j] = true;
                    chain.push(segments[j].0);
                    extended = true;
                    break;
                }
            }
            if !extended {
                break;
            }
        }
        loop {
            let mut extended = false;
            let head = chain[0];
            for j in 0..segments.len() {
                if used[j] {
                    continue;
                }
                if close_pt(head, segments[j].1) {
                    used[j] = true;
                    chain.insert(0, segments[j].0);
                    extended = true;
                    break;
                } else if close_pt(head, segments[j].0) {
                    used[j] = true;
                    chain.insert(0, segments[j].1);
                    extended = true;
                    break;
                }
            }
            if !extended {
                break;
            }
        }
        contours.push(chain);
    }

    Ok(contours)
}

/// Extracts a 3-D triangulated isosurface at `level` from a `[d, h, w]` volume using tetrahedral
/// marching-cubes decomposition (guaranteed manifold, crack-free, and free of ambiguous faces),
/// classifying active voxel cells on the device first.
pub fn marching_cubes<F: ImageFloat>(
    client: &Client,
    volume: &DeviceArray<F>,
    level: f64,
    spacing: Option<[f64; 3]>,
) -> OpsResult<SurfaceMesh> {
    check::ndim_in("marching_cubes", volume.shape(), &[3], "3")?;
    let (d, h, w) = (volume.dims()[0], volume.dims()[1], volume.dims()[2]);
    if d < 2 || h < 2 || w < 2 {
        return Ok(SurfaceMesh {
            vertices: Vec::new(),
            faces: Vec::new(),
            normals: Vec::new(),
        });
    }
    if !level.is_finite() {
        return Err(OpsError::InvalidParameter {
            name: "level",
            reason: "isosurface level must be finite".to_string(),
        });
    }
    let sp = spacing.unwrap_or([1.0, 1.0, 1.0]);
    if sp.iter().any(|&s| s <= 0.0 || !s.is_finite()) {
        return Err(OpsError::InvalidParameter {
            name: "spacing",
            reason: "voxel spacing must be positive and finite".to_string(),
        });
    }

    let (cd, ch, cw) = (d - 1, h - 1, w - 1);
    let num_cells = cd * ch * cw;
    let mask_dev = DeviceArray::<u32>::empty(client, Shape::new([cd, ch, cw]))?;
    let geom = LaunchGeometry::elementwise(client, num_cells);
    unsafe {
        marching_cubes_cell_mask_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            volume.binding(),
            mask_dev.binding(),
            h as u32,
            w as u32,
            ch as u32,
            cw as u32,
            cast::<F>(level),
            num_cells as u32,
        );
    }

    let masks = mask_dev.to_host(client);
    let vals: Vec<f64> = volume.to_host(client).into_iter().map(to_f64).collect();
    let slice = h * w;

    // 6-tetrahedra Kuhn decomposition sharing body diagonal (0 -> 6)
    const CORNERS: [[usize; 3]; 8] = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
    ];
    const TETS: [[usize; 4]; 6] = [
        [0, 5, 1, 6],
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
    ];
    // Tetrahedron edge table: for each 4-bit mask (0..16), up to 2 triangles given as pairs of
    // local tetrahedron vertex indices (0..4) for each of the 3 triangle vertices.
    #[allow(clippy::type_complexity)]
    const TET_TRIS: [&[([usize; 2], [usize; 2], [usize; 2])]; 16] = [
        &[],
        &[([0, 1], [0, 2], [0, 3])],
        &[([1, 0], [1, 3], [1, 2])],
        &[([0, 3], [1, 3], [1, 2]), ([0, 3], [1, 2], [0, 2])],
        &[([2, 0], [2, 1], [2, 3])],
        &[([0, 1], [2, 1], [2, 3]), ([0, 1], [2, 3], [0, 3])],
        &[([1, 0], [1, 3], [2, 3]), ([1, 0], [2, 3], [2, 0])],
        &[([3, 0], [3, 2], [3, 1])],
        &[([3, 0], [3, 1], [3, 2])],
        &[([0, 1], [0, 2], [3, 2]), ([0, 1], [3, 2], [3, 1])],
        &[([1, 0], [3, 0], [3, 2]), ([1, 0], [3, 2], [1, 2])],
        &[([2, 0], [2, 3], [2, 1])],
        &[([2, 0], [2, 1], [3, 1]), ([2, 0], [3, 1], [3, 0])],
        &[([1, 0], [1, 2], [1, 3])],
        &[([0, 1], [0, 3], [0, 2])],
        &[],
    ];

    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();

    let mut get_or_insert_edge = |g0: usize, g1: usize, p0: [f64; 3], p1: [f64; 3], v0: f64, v1: f64| -> usize {
        let key = if g0 < g1 { (g0, g1) } else { (g1, g0) };
        if let Some(&idx) = edge_map.get(&key) {
            return idx;
        }
        let denom = v1 - v0;
        let t = if denom.abs() <= 1e-12 {
            0.5
        } else {
            ((level - v0) / denom).clamp(0.0, 1.0)
        };
        let pos = [
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2]),
        ];
        let idx = vertices.len();
        vertices.push(pos);
        edge_map.insert(key, idx);
        idx
    };

    for z in 0..cd {
        for y in 0..ch {
            for x in 0..cw {
                let cmask = masks[z * (ch * cw) + y * cw + x];
                if cmask == 0 || cmask == 255 {
                    continue;
                }
                let mut g_idx = [0usize; 8];
                let mut c_pos = [[0.0f64; 3]; 8];
                let mut c_val = [0.0f64; 8];
                for k in 0..8 {
                    let gz = z + CORNERS[k][0];
                    let gy = y + CORNERS[k][1];
                    let gx = x + CORNERS[k][2];
                    let flat = gz * slice + gy * w + gx;
                    g_idx[k] = flat;
                    c_pos[k] = [(gz as f64) * sp[0], (gy as f64) * sp[1], (gx as f64) * sp[2]];
                    c_val[k] = vals[flat];
                }

                for tet in &TETS {
                    let mut tmask = 0usize;
                    for (bit, &ck) in tet.iter().enumerate() {
                        if c_val[ck] >= level {
                            tmask |= 1 << bit;
                        }
                    }
                    for &(e0, e1, e2) in TET_TRIS[tmask] {
                        let a0 = tet[e0[0]];
                        let a1 = tet[e0[1]];
                        let b0 = tet[e1[0]];
                        let b1 = tet[e1[1]];
                        let d0 = tet[e2[0]];
                        let d1 = tet[e2[1]];
                        let i0 = get_or_insert_edge(g_idx[a0], g_idx[a1], c_pos[a0], c_pos[a1], c_val[a0], c_val[a1]);
                        let i1 = get_or_insert_edge(g_idx[b0], g_idx[b1], c_pos[b0], c_pos[b1], c_val[b0], c_val[b1]);
                        let i2 = get_or_insert_edge(g_idx[d0], g_idx[d1], c_pos[d0], c_pos[d1], c_val[d0], c_val[d1]);
                        if i0 != i1 && i1 != i2 && i2 != i0 {
                            faces.push([i0, i1, i2]);
                        }
                    }
                }
            }
        }
    }

    // Compute area-weighted vertex normals
    let mut normals = vec![[0.0f64; 3]; vertices.len()];
    for &[i0, i1, i2] in &faces {
        let p0 = vertices[i0];
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &vi in &[i0, i1, i2] {
            normals[vi][0] += n[0];
            normals[vi][1] += n[1];
            normals[vi][2] += n[2];
        }
    }
    for n in &mut normals {
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if norm > 1e-12 {
            n[0] /= norm;
            n[1] /= norm;
            n[2] /= norm;
        }
    }

    Ok(SurfaceMesh { vertices, faces, normals })
}

/// Computes the total surface area of a triangulated 3-D [`SurfaceMesh`].
pub fn mesh_surface_area(mesh: &SurfaceMesh) -> f64 {
    let mut area = 0.0;
    for &[i0, i1, i2] in &mesh.faces {
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let p0 = mesh.vertices[i0];
        let p1 = mesh.vertices[i1];
        let p2 = mesh.vertices[i2];
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cx = u[1] * v[2] - u[2] * v[1];
        let cy = u[2] * v[0] - u[0] * v[2];
        let cz = u[0] * v[1] - u[1] * v[0];
        area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::runtime_test;

    fn marching_squares_and_cubes_extract_closed_contours_and_surfaces(client: &Client) {
        #[rustfmt::skip]
        let patch = vec![
            0.0f32, 0.0, 0.0, 0.0, 0.0,
            0.0,    1.0, 1.0, 1.0, 0.0,
            0.0,    1.0, 2.0, 1.0, 0.0,
            0.0,    1.0, 1.0, 1.0, 0.0,
            0.0,    0.0, 0.0, 0.0, 0.0,
        ];
        let arr2d = DeviceArray::from_host(client, Shape::new([5, 5]), patch).unwrap();
        let contours = marching_squares(client, &arr2d, 0.5).unwrap();
        assert_eq!(contours.len(), 1, "{}", client.name());
        assert!(contours[0].len() >= 8);

        // 5x5x5 volume with a high-valued 3x3x3 interior cube
        let mut vol = vec![0.0f32; 125];
        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    vol[z * 25 + y * 5 + x] = 1.0;
                }
            }
        }
        let arr3d = DeviceArray::from_host(client, Shape::new([5, 5, 5]), vol).unwrap();
        let mesh = marching_cubes(client, &arr3d, 0.5, None).unwrap();
        assert!(!mesh.vertices.is_empty(), "{}", client.name());
        assert!(!mesh.faces.is_empty(), "{}", client.name());
        let area = mesh_surface_area(&mesh);
        assert!(area > 20.0 && area < 60.0, "unexpected area {area}");
    }
    runtime_test!(test_contour, marching_squares_and_cubes_extract_closed_contours_and_surfaces);
}
