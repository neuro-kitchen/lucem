//! Kernel launch geometry: cube sizes from the runtime, cube counts from the problem.
//!
//! Every constructor of [`LaunchGeometry`] is paired with the `#[cube]` position helper its kernels
//! use. Kernels never compute positions from `ABSOLUTE_POS_X` / `CUBE_POS` themselves: the helpers
//! handle grid overflow and the CPU runtime's unit ordering (see [`fn@element_position`]).
//!
//! Ported from dsp-kitchen's `dspk-core/src/compute/launch.rs`, with image names: `rows_cols`
//! (was `channels_samples`), `per_col` (`per_sample`), `per_line` (`per_channel`).

use cubecl::prelude::*;

/// Kernel launch geometry for the access patterns used by lucem's kernels.
///
/// Cube sizes come from the runtime ([`CubeDim::new`]: its plane size along x, a plane count set by
/// the runtime's cores or limits along y), never from constants, and cube counts from the problem
/// size within the runtime's grid limits. CubeCL maps cubes and units to whatever executes the
/// kernel (CPU threads, GPU workgroups).
#[derive(Debug, Clone)]
pub struct LaunchGeometry {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
}

/// Element of the unit in a [`LaunchGeometry::elementwise`] launch; kernels bound-check it.
///
/// - **GPUs** (a cube is a plane or more wide): `ABSOLUTE_POS`, neighbouring units on neighbouring
///   elements, so a plane's reads coalesce.
/// - **Runtimes that run each unit of a cube on its own thread and loop it over every cube**
///   (CubeCL's CPU runtime: cubes one unit wide along x, one unit per core along y): unit `u` takes
///   the contiguous run `u · cubes .. (u + 1) · cubes`. With `ABSOLUTE_POS` the threads would
///   interleave element by element (`u, u + units, …`), sharing cache lines and leaving nothing
///   contiguous to vectorize: streaming kernels ran at 3 GB/s (dsp-kitchen, 2026-10-28, 12 cores).
///
/// Both cover `0..units · cubes`, so the results are the same; only the order differs.
#[cube]
pub fn element_position() -> usize {
    if CUBE_DIM_X == 1 && CUBE_DIM > 1 {
        UNIT_POS as usize * CUBE_COUNT + CUBE_POS
    } else {
        ABSOLUTE_POS
    }
}

/// Column of the unit in a [`LaunchGeometry::rows_cols`] or [`LaunchGeometry::per_col`] launch.
/// Columns run along x and continue along z when they need more cubes than one grid dimension
/// holds; kernels bound-check the result.
#[cube]
pub fn unit_col() -> u32 {
    (CUBE_POS_Z * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X
}

/// Row of the unit in a [`LaunchGeometry::rows_cols`] launch.
#[cube]
pub fn unit_row() -> u32 {
    ABSOLUTE_POS_Y
}

/// Line (row) walked by the unit in a [`LaunchGeometry::per_line`] launch; kernels bound-check it.
#[cube]
pub fn line_position() -> u32 {
    ABSOLUTE_POS_X
}

/// The columns `start..end` (every `step`-th) a unit walks in a [`LaunchGeometry::per_row`] cube of
/// `units` units over a row of `cols`:
///
/// - **GPUs**: `unit, unit + units, …`: neighbouring units on neighbouring columns (coalesced).
/// - **CPU runtime** (`PLANE_DIM == 1`, every unit its own thread): one contiguous run of
///   `⌈cols / units⌉` columns per unit, for the same reason as [`fn@element_position`].
///
/// Either way the units cover every column once, so per-unit partial results merge the same.
#[derive(CubeType)]
pub struct RowShare {
    pub start: u32,
    pub end: u32,
    pub step: u32,
}

/// This unit's [`RowShare`] of a row of `cols` columns.
#[cube]
pub fn row_share(cols: u32, #[comptime] units: u32) -> RowShare {
    let unit = UNIT_POS_X;
    // Field by field: a CubeCL `if` cannot yield a struct
    let contiguous = PLANE_DIM == 1u32;
    let chunk = cols.div_ceil(units);
    let start = if contiguous { u32::min(unit * chunk, cols) } else { unit };
    let end = if contiguous { u32::min(start + chunk, cols) } else { cols };
    let step = if contiguous { 1u32 } else { units };
    RowShare { start, end, step }
}

/// Row reduced by the cube in a [`LaunchGeometry::per_row`] launch (uniform across the cube), and the
/// tile index in a [`LaunchGeometry::tiles`] launch.
#[cube]
pub fn cube_row() -> u32 {
    CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X
}

/// Point of the unit in a [`LaunchGeometry::tiles`] launch; kernels bound-check it around their own
/// work (every unit takes part in the tile loads).
#[cube]
pub fn tile_point() -> u32 {
    cube_row() * CUBE_DIM_X + UNIT_POS_X
}

impl LaunchGeometry {
    /// One unit per element, indexed with [`fn@element_position`] (contiguous runs per thread on the
    /// CPU runtime, `ABSOLUTE_POS` elsewhere). The cubes are spread over the runtime's cube-count
    /// dimensions, so kernels must bound-check the position.
    pub fn elementwise(client: &Client, num_elements: usize) -> Self {
        let n = num_elements.max(1);
        let cube_dim = CubeDim::new(client, n);
        let cube_count = cubecl::calculate_cube_count_elemwise(client, n, cube_dim);
        Self { cube_dim, cube_count }
    }

    /// One unit per `(row, col)` of a `[rows, cols]` array: [`fn@unit_col`] and [`fn@unit_row`]. The
    /// cube's x size is the runtime's plane, so a plane walks neighbouring columns of one row (for
    /// kernels that need the 2-D position, e.g. a shared-memory tile with a halo).
    ///
    /// # Panics
    /// If the rows or columns need more cubes than the runtime's whole grid holds.
    pub fn rows_cols(client: &Client, rows: usize, cols: usize) -> Self {
        let cube_dim = CubeDim::new(client, rows.max(1) * cols.max(1));
        let (x, z) = Self::spill(client, cols, cube_dim.x);
        let y = (rows.max(1) as u32).div_ceil(cube_dim.y);
        assert!(y <= client.properties().hardware.max_cube_count.1, "{rows} rows exceed the runtime's cube grid");
        Self { cube_dim, cube_count: CubeCount::Static(x, y, z) }
    }

    /// One unit per column for kernels that walk all rows of a column ([`fn@unit_col`]).
    pub fn per_col(client: &Client, cols: usize) -> Self {
        let cube_dim = Self::flat(client, cols);
        let (x, z) = Self::spill(client, cols, cube_dim.x);
        Self { cube_dim, cube_count: CubeCount::Static(x, 1, z) }
    }

    /// One unit per line for kernels that walk each line sequentially (recursions, scans along a
    /// row): [`fn@line_position`], bound-checked by the kernel.
    pub fn per_line(client: &Client, lines: usize) -> Self {
        let cube_dim = Self::flat(client, lines);
        Self { cube_dim, cube_count: CubeCount::Static((lines.max(1) as u32).div_ceil(cube_dim.x), 1, 1) }
    }

    /// One cube per row of a `[rows, cols]` array, for cube-wide reductions: [`fn@cube_row`] is the
    /// row (cubes along x, continuing along y past the grid's x limit) and [`fn@row_share`] gives
    /// each unit its columns. The cube is the runtime's cube for `cols` units flattened onto x and
    /// rounded down to a power of two, so tree reductions can halve it; kernels size shared memory
    /// with `cube_dim.x` (passed as a comptime value). Kernels bound-check the row.
    ///
    /// # Panics
    /// If the rows need more cubes than the runtime's whole grid holds.
    pub fn per_row(client: &Client, rows: usize, cols: usize) -> Self {
        let units = Self::flat(client, cols).x;
        let cube_dim = CubeDim::new_1d(1 << (u32::BITS - 1 - units.max(1).leading_zeros()));
        let (x, y) = Self::spill(client, rows, 1);
        assert!(y <= client.properties().hardware.max_cube_count.1, "{rows} rows exceed the runtime's cube grid");
        Self { cube_dim, cube_count: CubeCount::Static(x, y, 1) }
    }

    /// One unit per point of `points`, in 1-D cubes of a power-of-two number of units (the runtime's
    /// cube flattened onto x, shrunk until `shared_bytes_per_unit` per unit fits the device's shared
    /// memory) for kernels that share tiles of points in shared memory: [`fn@tile_point`]; kernels
    /// size shared memory with `cube_dim.x` (passed as a comptime value).
    ///
    /// # Panics
    /// If one unit's shared memory exceeds the device's, or the points need more cubes than the
    /// runtime's whole grid holds.
    pub fn tiles(client: &Client, points: usize, shared_bytes_per_unit: usize) -> Self {
        let shared = client.properties().hardware.max_shared_memory_size;
        let fit = shared / shared_bytes_per_unit.max(1);
        assert!(fit >= 1, "{shared_bytes_per_unit} bytes of shared memory per unit exceed the device's {shared}");
        let units = Self::flat(client, points).x.min(u32::try_from(fit).unwrap_or(u32::MAX));
        let units = 1 << (u32::BITS - 1 - units.max(1).leading_zeros());
        let (x, y) = Self::spill(client, points, units);
        assert!(y <= client.properties().hardware.max_cube_count.1, "{points} points exceed the runtime's cube grid");
        Self { cube_dim: CubeDim::new_1d(units), cube_count: CubeCount::Static(x, y, 1) }
    }

    /// Units that execute in lock-step (the runtime's plane / warp / subgroup size; 1 where every unit
    /// runs on its own, as on the CPU runtime). It is the x size of [`Self::rows_cols`] cubes, so
    /// consecutive `unit_col`s of a plane share a cube.
    pub fn plane_lanes(client: &Client) -> u32 {
        client.properties().hardware.plane_size_max.max(1)
    }

    /// The runtime's cube for `work` units, flattened onto x (for kernels indexed along x only).
    fn flat(client: &Client, work: usize) -> CubeDim {
        CubeDim::new_1d(CubeDim::new(client, work.max(1)).num_elems())
    }

    /// Cube counts `(x, overflow)` covering `units` along x with `per_cube` units per cube; the
    /// overflow count continues past the grid's x limit (along z for columns, y for rows).
    fn spill(client: &Client, units: usize, per_cube: u32) -> (u32, u32) {
        let max = client.properties().hardware.max_cube_count;
        let cubes = (units.max(1) as u64).div_ceil(per_cube as u64);
        let x = cubes.min(max.0 as u64);
        let overflow = cubes.div_ceil(x);
        assert!(overflow <= max.2.min(max.1) as u64, "{units} units exceed the runtime's cube grid");
        (x as u32, overflow as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubes(count: &CubeCount) -> u64 {
        match count {
            CubeCount::Static(x, y, z) => *x as u64 * *y as u64 * *z as u64,
            _ => unreachable!(),
        }
    }

    #[test]
    fn geometries_cover_their_ranges_within_limits() {
        let targets = crate::ComputeTarget::for_tests();
        assert!(!targets.is_empty(), "no CubeCL runtime to test (compiled in and allowed by LUCEM_TEST_RUNTIMES)");
        for target in targets {
            geometries_cover(&target.client().expect("compiled-in runtime"));
        }
    }

    fn geometries_cover(client: &Client) {
        let max = client.properties().hardware.max_cube_count;
        let within = |c: &CubeCount| matches!(c, CubeCount::Static(x, y, z) if *x <= max.0 && *y <= max.1 && *z <= max.2);

        // A 3-channel 16k × 16k image: more cubes than one dimension allows
        let n = 3 * 16_384 * 16_384;
        let geom = LaunchGeometry::elementwise(client, n);
        assert!(within(&geom.cube_count) && cubes(&geom.cube_count) * geom.cube_dim.num_elems() as u64 >= n as u64);

        // Columns beyond one grid dimension spill into z
        for cols in [30_001usize, 5_000_000] {
            let geom = LaunchGeometry::rows_cols(client, 385, cols);
            let CubeCount::Static(x, y, z) = geom.cube_count else { unreachable!() };
            assert!(within(&geom.cube_count));
            assert!(x as u64 * z as u64 * geom.cube_dim.x as u64 >= cols as u64 && y * geom.cube_dim.y >= 385);
            assert_eq!(geom.cube_dim.x, LaunchGeometry::plane_lanes(client), "{}", client.name());
        }
        let geom = LaunchGeometry::per_line(client, 385);
        assert!(cubes(&geom.cube_count) * geom.cube_dim.num_elems() as u64 >= 385);
    }
}
