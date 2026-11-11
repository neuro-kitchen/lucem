---
title: "lumen_ops::register"
description: "Image registration & dense optical flow: subpixel Fourier phase cross-correlation (`phase_correlation`) and dense 2-D optical flow (`ilk_flow`, `tvl1_flow`)."
sidebar:
  label: "register"
  order: 11
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Image registration & dense optical flow: subpixel Fourier phase cross-correlation
(`phase_correlation`) and dense 2-D optical flow (`ilk_flow`, `tvl1_flow`).

## Types

### `OpticalFlow2d`

```rust
pub struct OpticalFlow2d<F: ImageFloat>
```

Dense 2-D displacement vector field `(vy, vx)` mapping coordinates `(y, x)` in `reference` to
`(y + vy[y, x], x + vx[y, x])` in `moving`.

**Fields**

- `vy: DeviceArray<F>`: Vertical (row / axis 0) displacement component.
- `vx: DeviceArray<F>`: Horizontal (column / axis 1) displacement component.

Implements `Clone`, `Debug`.

### `IlkFlow`

```rust
pub struct IlkFlow
```

Settings for Iterative Lucas-Kanade dense optical flow (`ilk_flow`).

**Fields**

- `radius: usize`: Neighborhood integration radius (`window_size = 2 * radius + 1`).
- `num_warp: usize`: Number of iterative warping refinements.
- `gaussian: bool`: When `true`, uses a Gaussian integration window ($\sigma = \text{radius} / 2$); when `false`, uses a uniform `(2 * radius + 1)` box window.

Implements `Clone`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `Tvl1Flow`

```rust
pub struct Tvl1Flow
```

Settings for Zach, Pock & Bischof's (2007) Dual TV-$L_1$ dense optical flow (`tvl1_flow`).

**Fields**

- `attachment: f64`: Data attachment weight $\lambda > 0$ (larger values follow image gradients more closely).
- `tightness: f64`: Coupling tightness $\theta > 0$ between primal flow $u$ and auxiliary flow $v$.
- `num_warp: usize`: Number of outer warping linearizations.
- `num_iter: usize`: Number of inner primal-dual TV-$L_1$ iterations per warp.

Implements `Clone`, `Debug`, `Default`, `PartialEq`.

### `PhaseCorrelation`

```rust
pub struct PhaseCorrelation
```

Settings for `phase_correlation`.

**Fields**

- `upsample_factor: usize`: Subpixel upsampling factor (`1` for integer-pixel registration; `> 1` for subpixel accuracy of `1.0 / upsample_factor` pixels).
- `normalize_phase: bool`: When `true`, normalizes the cross-power spectrum by its pointwise magnitude (`R / |R|`, phase correlation); when `false`, uses unnormalized cross-correlation (`R = F_ref * conj(F_mov)`).

Implements `Clone`, `Debug`, `Default`, `Eq`, `PartialEq`.

### `PhaseCorrelationResult`

```rust
pub struct PhaseCorrelationResult
```

Output of `phase_correlation`.

**Fields**

- `shift: Vec<f64>`: Shift vector (in pixels along each axis) required to register `moving` with `reference`.
- `error: f64`: Translation-invariant normalized RMS registration error in `[0.0, 1.0]`.
- `phasediff: f64`: Global phase difference (in radians, `[-pi, pi]`) between the two images at the peak.

Implements `Clone`, `Debug`, `PartialEq`.

## Functions

### `ilk_flow_into`

```rust
pub fn ilk_flow_into<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    vy_out: &DeviceArray<F>,
    vx_out: &DeviceArray<F>,
    settings: &IlkFlow,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Estimates dense 2-D optical flow `(vy, vx)` from `reference` (`image0`) to `moving` (`image1`)
using the Iterative Lucas-Kanade (ILK) solver on the device.

### `ilk_flow`

```rust
pub fn ilk_flow<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &IlkFlow,
) -> OpsResult<OpticalFlow2d<F>>
```

Allocates and returns dense 2-D Iterative Lucas-Kanade optical flow `(vy, vx)`.

### `tvl1_flow_into`

```rust
pub fn tvl1_flow_into<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    vy_out: &DeviceArray<F>,
    vx_out: &DeviceArray<F>,
    settings: &Tvl1Flow,
    ws: &mut Workspace,
) -> OpsResult<()>
```

Estimates dense 2-D optical flow `(vy, vx)` from `reference` to `moving` using the Dual TV-$L_1$
variational solver on the device.

### `tvl1_flow`

```rust
pub fn tvl1_flow<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &Tvl1Flow,
) -> OpsResult<OpticalFlow2d<F>>
```

Allocates and returns dense 2-D Dual TV-$L_1$ optical flow `(vy, vx)`.

### `phase_correlation`

```rust
pub fn phase_correlation<F: ImageFloat>(
    client: &Client,
    reference: &DeviceArray<F>,
    moving: &DeviceArray<F>,
    settings: &PhaseCorrelation,
) -> OpsResult<PhaseCorrelationResult>
```

Estimates the N-D translation shift (with optional `1 / upsample_factor` subpixel precision) that
registers `moving` onto `reference`.
