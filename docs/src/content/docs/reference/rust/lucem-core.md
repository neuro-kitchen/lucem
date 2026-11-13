---
title: "lucem_core"
description: "Device arrays and memory, shapes, compute targets (runtimes), tiling."
sidebar:
  label: "lucem-core"
  order: 1
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Core of lucem: the one place every crate reaches a GPU (or the CPU runtime) through.

- `device`: `ComputeTarget`, the name of a CubeCL runtime across the workspace (CLI flag,
  Python argument, app setting, `LUCEM_RUNTIME`). Needs no GPU libraries.
- `shape`: `Shape` of C-contiguous arrays and `AxisSplit`, the `[outer, len, inner]` view of
  one axis that lets a 1-D kernel work along any axis.
- `compute` (`compute` feature): clients, persisted kernel and autotune caches, device shutdown
  at exit, launch geometry, autotune helpers, timing.
- `memory` (`compute` feature): `DeviceArray`, typed buffers,
  `Workspace` and `DeviceCache` for reusing device
  memory, and the memory budget.
- `element` (`compute` feature): the element types kernels are generic over.

Ported from dsp-kitchen's `dspk-core` (compute layer) and `dspk-ops::core` (buffers); the reasons
behind each rule are in the maintainers' notes (`neuro-kitchen/knowledge/cubecl/`, outside this repository).

## Types

### `Shape`

```rust
pub struct Shape
```

Shape of a C-contiguous array. A shape with no dimensions is a scalar (one element).

**Fields**

- `(_)`

**Methods**

```rust
pub fn new(dims: impl Into<Vec<usize>>) -> Self
```

```rust
pub fn vector(len: usize) -> Self
```

A 1-D shape of `len` elements.

```rust
pub fn dims(&self) -> &[usize]
```

```rust
pub fn ndim(&self) -> usize
```

```rust
pub fn len(&self) -> usize
```

Number of elements.

```rust
pub fn is_empty(&self) -> bool
```

```rust
pub fn axis(&self, axis: isize) -> CoreResult<usize>
```

`axis` normalized as numpy does (negative counts from the end).

```rust
pub fn axes(&self, axes: Option<&[isize]>) -> CoreResult<Vec<usize>>
```

`axes` normalized (numpy rules), every axis at most once; `None` means every axis.

```rust
pub fn axis_split(&self, axis: usize) -> AxisSplit
```

The `[outer, len, inner]` view of `axis` (already normalized).

#### Panics
If `axis >= ndim`.

```rust
pub fn strides(&self) -> Vec<usize>
```

C-order strides in elements.

```rust
pub fn expect(&self, other: &Shape) -> CoreResult<()>
```

`Ok` if `self` equals `other`, else `CoreError::ShapeMismatch`.

Implements `Clone`, `Debug`, `Default`, `Eq`, `From`, `Hash`, `PartialEq`.

### `AxisSplit`

```rust
pub struct AxisSplit
```

One axis of a C-contiguous array seen as `[outer, len, inner]`: element `(o, i, k)` sits at
`(o · len + i) · inner + k`. Lets one 1-D kernel work along any axis of any rank: neighbouring
units differ in `k` (or in `i` when `inner == 1`), so reads coalesce on every axis.

**Fields**

- `outer: usize`: Product of the dimensions before the axis.
- `len: usize`: Length of the axis.
- `inner: usize`: Product of the dimensions after the axis (the axis's stride in elements).

Implements `Clone`, `Copy`, `Debug`, `Eq`, `PartialEq`.

### `Block`

```rust
pub struct Block
```

One step of a walk over an image.

**Fields**

- `index: usize`: Position in the walk.
- `origin: Zyx`: The part this step owns.
- `size: Zyx`
- `read_origin: Zyx`: The part read for it: the owned part plus the halo, clipped to the image.
- `read_size: Zyx`

**Methods**

```rust
pub fn inner_offset(&self) -> Zyx
```

Where the owned part starts inside the read part.

```rust
pub fn read_len(&self) -> usize
```

Pixels the step reads.

```rust
pub fn len(&self) -> usize
```

Pixels the step owns.

```rust
pub fn is_empty(&self) -> bool
```

```rust
pub fn read_dims(&self) -> Vec<usize>
```

The array dimensions of the read part: `[h, w]` when one plane deep, else `[d, h, w]`.

Implements `Clone`, `Copy`, `Debug`, `Eq`, `Hash`, `PartialEq`.

### `ComputeTarget`

```rust
pub enum ComputeTarget
```

A CubeCL runtime and its default device.

**Variants**

- `Wgpu`: WebGPU (Vulkan / Metal / DX12).
- `Cpu`: Multi-threaded CPU (MLIR JIT).
- `Cuda`: NVIDIA CUDA.
- `Hip`: AMD ROCm HIP.

**Methods**

```rust
pub fn name(self) -> &'static str
```

```rust
pub fn parse(name: &str) -> Result<Self, ComputeError>
```

The runtime called `name` (case-insensitive; `vulkan`, `metal`, `gpu` → wgpu, `rocm` → hip).

```rust
pub fn device(self) -> Result<Device, ComputeError>
```

The default device of this runtime, if it is compiled in.

```rust
pub fn is_available(self) -> bool
```

Whether this runtime is compiled in.

```rust
pub fn client(self) -> Result<Client, ComputeError>
```

A client for this runtime's default device (recorded for `shutdown_devices`).

```rust
pub fn available() -> Vec<Self>
```

Compiled-in runtimes in preference order (GPU runtimes first).

```rust
pub fn run<T: ComputeTask>(self, task: T) -> Result<<T as >::Output, ComputeError>
```

Runs `task` with a client for this runtime's default device.

```rust
pub fn is_usable(self) -> bool
```

Compiled in and its driver there. Opening CUDA or HIP without the driver panics on the
runtime's device thread and takes the caller down, so they are asked of the system first:
the NVIDIA kernel module (`/proc/driver/nvidia/version`, Linux) or AMD's (`/dev/kfd`).

```rust
pub fn unusable_reason(self) -> Option<&'static str>
```

Why a compiled-in runtime cannot run here (`None`: it can, as far as the system tells).

```rust
pub fn usable() -> Vec<Self>
```

Usable runtimes in preference order.

```rust
pub fn for_tests() -> Vec<Self>
```

The runtimes tests run on: `Self::available`, limited to the names in `LUCEM_TEST_RUNTIMES`
when it is set. Compiled in is not the same as usable: a build with CUDA can run on a
machine without an NVIDIA driver.

#### Panics
If `LUCEM_TEST_RUNTIMES` names an unknown runtime (a typo would silently skip tests).

```rust
pub fn from_env() -> Result<Self, ComputeError>
```

`LUCEM_RUNTIME` when set, else the first usable runtime.

```rust
pub fn checked(self) -> Result<Self, ComputeError>
```

`self` if it is compiled in and its driver is there, else an error saying which.

Implements `Clone`, `Copy`, `Debug`, `Deserialize`, `Display`, `Eq`, `FromStr`, `Hash`, `PartialEq`, `Serialize`.

### `ComputeError`

```rust
pub enum ComputeError
```

**Variants**

- `Unknown(String)`
- `Unavailable(ComputeTarget, &'static str)`
- `NoneAvailable`
- `NoDriver(ComputeTarget, &'static str)`

Implements `Clone`, `Debug`, `Display`, `Eq`, `Error`, `From`, `PartialEq`.

### `CoreError`

```rust
pub enum CoreError
```

**Variants**

- `ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> }`: Two shapes that must agree do not.
- `AxisOutOfBounds { axis: isize, ndim: usize }`: An axis outside `-ndim..ndim` (numpy's message).
- `RepeatedAxis { axis: isize }`: The same axis given twice (numpy's message).
- `TooManyElements { what: String, dims: Vec<usize>, limit: usize }`: More elements than one device buffer may hold (kernels index with 32-bit integers).
- `OutOfDeviceMemory { what: String, requested: u64, available: u64, limit: u64 }`: An allocation larger than the device memory budget allows.
- `BufferTooSmall { what: String, requested: u64, capacity: u64 }`: A buffer smaller than the data written into or read from it.
- `Compute(ComputeError)`: A compute runtime could not be selected or opened.

Implements `Clone`, `Debug`, `Display`, `Error`, `From`, `PartialEq`.

### `CoreResult`

```rust
pub type CoreResult<T> = Result<T, CoreError>;
```

### `Zyx`

```rust
pub type Zyx = [usize; 3];
```

`[z, y, x]`.

## `lucem_core::compute`

The compute layer (`compute` feature): the single entry point through which every crate reaches a
CubeCL runtime.

### `LaunchGeometry`

```rust
pub struct LaunchGeometry
```

Kernel launch geometry for the access patterns used by lucem's kernels.

Cube sizes come from the runtime (`CubeDim::new`: its plane size along x, a plane count set by
the runtime's cores or limits along y), never from constants, and cube counts from the problem
size within the runtime's grid limits. CubeCL maps cubes and units to whatever executes the
kernel (CPU threads, GPU workgroups).

**Fields**

- `cube_dim: CubeDim`
- `cube_count: CubeCount`

**Methods**

```rust
pub fn elementwise(client: &Client, num_elements: usize) -> Self
```

One unit per element, indexed with `fn@element_position` (contiguous runs per thread on the
CPU runtime, `ABSOLUTE_POS` elsewhere). The cubes are spread over the runtime's cube-count
dimensions, so kernels must bound-check the position.

```rust
pub fn rows_cols(client: &Client, rows: usize, cols: usize) -> Self
```

One unit per `(row, col)` of a `[rows, cols]` array: `fn@unit_col` and `fn@unit_row`. The
cube's x size is the runtime's plane, so a plane walks neighbouring columns of one row (for
kernels that need the 2-D position, e.g. a shared-memory tile with a halo).

#### Panics
If the rows or columns need more cubes than the runtime's whole grid holds.

```rust
pub fn per_col(client: &Client, cols: usize) -> Self
```

One unit per column for kernels that walk all rows of a column (`fn@unit_col`).

```rust
pub fn per_line(client: &Client, lines: usize) -> Self
```

One unit per line for kernels that walk each line sequentially (recursions, scans along a
row): `fn@line_position`, bound-checked by the kernel.

```rust
pub fn per_row(client: &Client, rows: usize, cols: usize) -> Self
```

One cube per row of a `[rows, cols]` array, for cube-wide reductions: `fn@cube_row` is the
row (cubes along x, continuing along y past the grid's x limit) and `fn@row_share` gives
each unit its columns. The cube is the runtime's cube for `cols` units flattened onto x and
rounded down to a power of two, so tree reductions can halve it; kernels size shared memory
with `cube_dim.x` (passed as a comptime value). Kernels bound-check the row.

#### Panics
If the rows need more cubes than the runtime's whole grid holds.

```rust
pub fn tiles(client: &Client, points: usize, shared_bytes_per_unit: usize) -> Self
```

One unit per point of `points`, in 1-D cubes of a power-of-two number of units (the runtime's
cube flattened onto x, shrunk until `shared_bytes_per_unit` per unit fits the device's shared
memory) for kernels that share tiles of points in shared memory: `fn@tile_point`; kernels
size shared memory with `cube_dim.x` (passed as a comptime value).

#### Panics
If one unit's shared memory exceeds the device's, or the points need more cubes than the
runtime's whole grid holds.

```rust
pub fn plane_lanes(client: &Client) -> u32
```

Units that execute in lock-step (the runtime's plane / warp / subgroup size; 1 where every unit
runs on its own, as on the CPU runtime). It is the x size of `Self::rows_cols` cubes, so
consecutive `unit_col`s of a plane share a cube.

Implements `Clone`, `Debug`.

### `RowShare`

```rust
pub struct RowShare
```

The columns `start..end` (every `step`-th) a unit walks in a `LaunchGeometry::per_row` cube of
`units` units over a row of `cols`:

- **GPUs**: `unit, unit + units, …`: neighbouring units on neighbouring columns (coalesced).
- **CPU runtime** (`PLANE_DIM == 1`, every unit its own thread): one contiguous run of
  `⌈cols / units⌉` columns per unit, for the same reason as `fn@element_position`.

Either way the units cover every column once, so per-unit partial results merge the same.

**Fields**

- `start: u32`
- `end: u32`
- `step: u32`

Implements `CubeType`.

### `PinnedChoices`

```rust
pub struct PinnedChoices
```

While alive, `tuned_choices_pinned` is `true` on this thread. Guards nest.

Implements `Drop`.

### `ComputeTask`

```rust
pub trait ComputeTask
```

Work to run on whichever runtime a `ComputeTarget` selects.

**Methods**

```rust
fn run(self, client: Client) -> <Self as >::Output;
```

### `element_position`

```rust
pub fn element_position() -> usize
```

Element of the unit in a `LaunchGeometry::elementwise` launch; kernels bound-check it.

- **GPUs** (a cube is a plane or more wide): `ABSOLUTE_POS`, neighbouring units on neighbouring
  elements, so a plane's reads coalesce.
- **Runtimes that run each unit of a cube on its own thread and loop it over every cube**
  (CubeCL's CPU runtime: cubes one unit wide along x, one unit per core along y): unit `u` takes
  the contiguous run `u · cubes .. (u + 1) · cubes`. With `ABSOLUTE_POS` the threads would
  interleave element by element (`u, u + units, …`), sharing cache lines and leaving nothing
  contiguous to vectorize: streaming kernels ran at 3 GB/s (dsp-kitchen, 2026-10-28, 12 cores).

Both cover `0..units · cubes`, so the results are the same; only the order differs.

### `unit_col`

```rust
pub fn unit_col() -> u32
```

Column of the unit in a `LaunchGeometry::rows_cols` or `LaunchGeometry::per_col` launch.
Columns run along x and continue along z when they need more cubes than one grid dimension
holds; kernels bound-check the result.

### `unit_row`

```rust
pub fn unit_row() -> u32
```

Row of the unit in a `LaunchGeometry::rows_cols` launch.

### `line_position`

```rust
pub fn line_position() -> u32
```

Line (row) walked by the unit in a `LaunchGeometry::per_line` launch; kernels bound-check it.

### `row_share`

```rust
pub fn row_share(cols: u32, units: u32) -> RowShare
```

This unit's `RowShare` of a row of `cols` columns.

### `cube_row`

```rust
pub fn cube_row() -> u32
```

Row reduced by the cube in a `LaunchGeometry::per_row` launch (uniform across the cube), and the
tile index in a `LaunchGeometry::tiles` launch.

### `tile_point`

```rust
pub fn tile_point() -> u32
```

Point of the unit in a `LaunchGeometry::tiles` launch; kernels bound-check it around their own
work (every unit takes part in the tile loads).

### `device_elements`

```rust
pub fn device_elements(what: &str, dims: &[usize]) -> CoreResult<usize>
```

The product of `dims` as an element count, or an error naming `what` when it exceeds
`MAX_DEVICE_ELEMENTS` (or overflows).

### `pin_tuned_choices`

```rust
pub fn pin_tuned_choices() -> PinnedChoices
```

Pins result-changing autotuned choices on this thread until the guard is dropped.

### `tuned_choices_pinned`

```rust
pub fn tuned_choices_pinned() -> bool
```

Whether a `PinnedChoices` guard is alive on this thread.

### `positive_infinity`

```rust
pub fn positive_infinity<F: Float>() -> F
```

+∞ in `F`.

### `negative_infinity`

```rust
pub fn negative_infinity<F: Float>() -> F
```

−∞ in `F`.

### `configure_runtime`

```rust
pub fn configure_runtime()
```

Sets CubeCL's runtime configuration once, before the first device opens: compiled kernels are
cached (CubeCL's default is off), and the caches live in the user cache directory (or
`LUCEM_CACHE_DIR`) instead of the `target/` of whichever Cargo project the process runs in.
A `cubecl.toml` or CubeCL's own environment variables still apply. Called by
`ComputeTarget::device` and `open_device`; harmless to call again.

### `open_device`

```rust
pub fn open_device(device: Device) -> Client
```

The client of `device`, recorded so that `shutdown_devices` shuts it down at exit. Use this
instead of `Device::client` wherever a specific device is opened (e.g. one GPU of several).

### `shutdown_devices`

```rust
pub fn shutdown_devices()
```

Shuts down every device a client was made for: their runner threads stop (queued work runs
first) and the devices are destroyed, so the GPU driver's threads end before the process does.
Runs at exit by itself; call it earlier to release the GPUs. Clients still alive keep their
runner going (CubeCL waits up to 30 s for it, then leaves it), so drop them first. A later
`ComputeTarget::client` opens the device again.

### `MAX_DEVICE_ELEMENTS`

```rust
pub const MAX_DEVICE_ELEMENTS: usize = _;
```

Most elements one device buffer may hold (`u32::MAX`).

### `POSITIVE_INFINITY_BITS`

```rust
pub const POSITIVE_INFINITY_BITS: u32 = _;
```

IEEE-754 bits of `f32` +∞.

### `NEGATIVE_INFINITY_BITS`

```rust
pub const NEGATIVE_INFINITY_BITS: u32 = _;
```

IEEE-754 bits of `f32` −∞.

### `KERNEL_CACHE_ENV`

```rust
pub const KERNEL_CACHE_ENV: &str = "LUCEM_KERNEL_CACHE";
```

Environment variable: `0` turns off the compiled-kernel cache (autotune decisions still persist).

### `CACHE_DIR_ENV`

```rust
pub const CACHE_DIR_ENV: &str = "LUCEM_CACHE_DIR";
```

Environment variable: directory for CubeCL's caches (autotune decisions, compiled kernels);
default: the user cache directory.

### `RUNTIME_ENV`

```rust
pub const RUNTIME_ENV: &str = "LUCEM_RUNTIME";
```

Environment variable that overrides `ComputeTarget::from_env`'s default (`wgpu`, `cpu`,
`cuda`, `hip`).

### `TEST_RUNTIMES_ENV`

```rust
pub const TEST_RUNTIMES_ENV: &str = "LUCEM_TEST_RUNTIMES";
```

Environment variable that limits the runtimes tests run on (`ComputeTarget::for_tests`):
comma-separated names, e.g. `cpu` on a CI machine that has CUDA compiled in but no GPU.

## `lucem_core::compute::bench`

Device-synchronised timing for kernel benchmarks.

### `sync`

```rust
pub fn sync(client: &Client)
```

Waits until every operation queued on `client` has finished.

### `time_device`

```rust
pub fn time_device(client: &Client, iters: usize, f: impl FnMut()) -> Duration
```

Runs `f` once as warm-up (kernel compilation, allocation), then `iters` times, and returns the
median wall time per iteration including device completion.

## `lucem_core::compute::launch`

Kernel launch geometry: cube sizes from the runtime, cube counts from the problem.

### `RowShareExpand`

```rust
pub struct RowShareExpand
```

**Fields**

- `start: <u32 as CubeType>::ExpandType`
- `end: <u32 as CubeType>::ExpandType`
- `step: <u32 as CubeType>::ExpandType`

Implements `AsMutExpand`, `AsRefExpand`, `CubeDebug`, `ExpandTypeClone`, `IntoExpand`, `IntoMut`.

## `lucem_core::compute::launch::cube_row`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <u32 as CubeType>::ExpandType
```

Row reduced by the cube in a `LaunchGeometry::per_row` launch (uniform across the cube), and the
tile index in a `LaunchGeometry::tiles` launch.

## `lucem_core::compute::launch::element_position`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <usize as CubeType>::ExpandType
```

Element of the unit in a `LaunchGeometry::elementwise` launch; kernels bound-check it.

- **GPUs** (a cube is a plane or more wide): `ABSOLUTE_POS`, neighbouring units on neighbouring
  elements, so a plane's reads coalesce.
- **Runtimes that run each unit of a cube on its own thread and loop it over every cube**
  (CubeCL's CPU runtime: cubes one unit wide along x, one unit per core along y): unit `u` takes
  the contiguous run `u · cubes .. (u + 1) · cubes`. With `ABSOLUTE_POS` the threads would
  interleave element by element (`u, u + units, …`), sharing cache lines and leaving nothing
  contiguous to vectorize: streaming kernels ran at 3 GB/s (dsp-kitchen, 2026-10-28, 12 cores).

Both cover `0..units · cubes`, so the results are the same; only the order differs.

## `lucem_core::compute::launch::line_position`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <u32 as CubeType>::ExpandType
```

Line (row) walked by the unit in a `LaunchGeometry::per_line` launch; kernels bound-check it.

## `lucem_core::compute::launch::row_share`

### `expand`

```rust
pub fn expand(
    scope: &Scope,
    cols: <u32 as CubeType>::ExpandType,
    units: u32,
) -> <RowShare as CubeType>::ExpandType
```

This unit's `RowShare` of a row of `cols` columns.

## `lucem_core::compute::launch::tile_point`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <u32 as CubeType>::ExpandType
```

Point of the unit in a `LaunchGeometry::tiles` launch; kernels bound-check it around their own
work (every unit takes part in the tile loads).

## `lucem_core::compute::launch::unit_col`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <u32 as CubeType>::ExpandType
```

Column of the unit in a `LaunchGeometry::rows_cols` or `LaunchGeometry::per_col` launch.
Columns run along x and continue along z when they need more cubes than one grid dimension
holds; kernels bound-check the result.

## `lucem_core::compute::launch::unit_row`

### `expand`

```rust
pub fn expand(scope: &Scope) -> <u32 as CubeType>::ExpandType
```

Row of the unit in a `LaunchGeometry::rows_cols` launch.

## `lucem_core::compute::tune`

Helpers for CubeCL autotuning of launch settings.

### `tune_id`

```rust
pub fn tune_id(client: &Client) -> String
```

Autotune id of the device behind `client`: runtime name plus the hardware properties that shape
launches, so different devices keep separate tuning results.

### `size_class`

```rust
pub fn size_class(n: usize) -> usize
```

Problem sizes grouped by power of two, so similar sizes share one tuning result.

## `lucem_core::element`

Element types kernels are generic over.

### `ImageFloat`

```rust
pub trait ImageFloat
```

A CubeCL float that also crosses to and from the host and can be a scalar kernel argument
(`LaunchArg<RuntimeArg = Self>`; `Float` alone is not enough for a scalar argument).

### `ImageInt`

```rust
pub trait ImageInt
```

A CubeCL integer that crosses to and from the host and can be a scalar kernel argument (labels,
counts, stored pixel types such as `u8` / `u16` / `u32`).

### `cast`

```rust
pub fn cast<F: ImageFloat>(value: f64) -> F
```

`value` as `F` (rounded to its precision). Host constants enter as `f64` and are cast once.

#### Panics
If `value` is not representable in `F` (e.g. NaN in a type without it, or out of range).

### `cast_all`

```rust
pub fn cast_all<F: ImageFloat>(values: &[f64]) -> Vec<F>
```

`values` as `F`.

### `to_f64`

```rust
pub fn to_f64<F: ImageFloat>(value: F) -> f64
```

`value` as `f64`.

### `supports`

```rust
pub fn supports<E: CubeElement + Scalar>(client: &Client) -> bool
```

Whether the runtime behind `client` supports `E` (e.g. `f64` is missing on most wgpu adapters).

## `lucem_core::memory`

Device memory: arrays that stay on the device, and reuse of what was allocated.

### `DeviceArray`

```rust
pub struct DeviceArray<E: CubeElement>
```

A C-contiguous array of `E` on one device: what every lucem operation takes and returns.

Data moves between host and device only through `Self::from_host` / `Self::write_from_host`
and `Self::to_host` / `Self::to_host_range`; everything in between stays on the device.

Cloning is cheap and **shares the buffer** (as Burn tensors do): never write into an array you
did not allocate or take from a `Workspace`. The handle may address more
bytes than the array uses (a reused buffer); `Self::binding` covers exactly `len` elements.

**Methods**

```rust
pub fn empty(client: &Client, shape: Shape) -> CoreResult<Self>
```

Uninitialized array. Checks the 32-bit index limit and, for large arrays, the memory budget.

```rust
pub fn from_host(client: &Client, shape: Shape, data: Vec<E>) -> CoreResult<Self>
```

Device copy of `data` (C order), moving the bytes into the transfer (the fast upload path).

```rust
pub fn from_slice(client: &Client, shape: Shape, data: &[E]) -> CoreResult<Self>
```

Device copy of `data` (copies the slice once on the host, then as `Self::from_host`).

```rust
pub fn from_handle(handle: Handle, shape: Shape) -> Self
```

An array over an existing buffer that holds at least `shape.len()` elements of `E` (from a
`Scratch`, `Workspace` or `DeviceCache`).

#### Panics
If the buffer is smaller than the shape needs.

```rust
pub fn write_from_host(&self, client: &Client, data: Vec<E>) -> CoreResult<()>
```

Writes `data` (C order, exactly `len` elements) into this array's buffer: no allocation,
queued after the work already on the stream (an array refilled for every image of a batch).

```rust
pub fn to_host(&self, client: &Client) -> Vec<E>
```

Host copy of the whole array (waits for the queued work that writes it).

```rust
pub fn to_host_range(&self, client: &Client, start: usize, len: usize) -> Vec<E>
```

Host copy of elements `start .. start + len` (C order): only those bytes are transferred.

#### Panics
If the range exceeds the array.

```rust
pub fn binding(&self) -> BufferArg
```

The kernel argument for this array: a binding of exactly `len` elements.

```rust
pub fn reshaped(&self, shape: Shape) -> CoreResult<Self>
```

Same buffer, other shape with the same number of elements (a free reshape).

```rust
pub fn shares_buffer_with<T: CubeElement>(&self, other: &DeviceArray<T>) -> bool
```

Whether `self` and `other` use the same device buffer (kernels must not bind both: no aliased
bindings). Clones of a handle share one memory descriptor, so pointer identity decides.

```rust
pub fn shape(&self) -> &Shape
```

```rust
pub fn dims(&self) -> &[usize]
```

```rust
pub fn ndim(&self) -> usize
```

```rust
pub fn len(&self) -> usize
```

```rust
pub fn is_empty(&self) -> bool
```

```rust
pub fn handle(&self) -> &Handle
```

```rust
pub fn into_handle(self) -> Handle
```

```rust
pub fn filled(client: &Client, shape: Shape, value: F) -> CoreResult<Self>
```

Array of `value`, filled on the device (nothing uploaded).

```rust
pub fn zeros(client: &Client, shape: Shape) -> CoreResult<Self>
```

Array of zeros, filled on the device (nothing uploaded).

Implements `Clone`, `Debug`.

### `DeviceCache`

```rust
pub struct DeviceCache
```

Constant device data (filter taps, footprint offsets, colour matrices, look-up tables) uploaded
once per device and key, then reused by every call.

The key always includes the device: a handle used on another device is not an error but garbage
(CorridorKey once reused a CUDA handle on wgpu through a cache keyed by shape alone). Keys name
everything that changes the data, including the element type (e.g. `"f32-sigma2-order0-r8"`).

Declare one per kind of data as a static: `static TAPS: DeviceCache = DeviceCache::new();`.
Entries live for the process; a cache only holds small constants.

**Methods**

```rust
pub const fn new() -> Self
```

```rust
pub fn get_or_upload<E: CubeElement>(
    &self,
    client: &Client,
    key: &str,
    shape: Shape,
    make: impl FnOnce() -> Vec<E>,
) -> DeviceArray<E>
```

The array stored under `key` for `client`'s device, uploading `make()` (C order, `shape`) the
first time.

#### Panics
If `make()` returns a different number of elements than `shape` holds.

```rust
pub fn clear_device(&self, client: &Client)
```

Drops every entry of `client`'s device (e.g. before shutting the device down).

```rust
pub fn len(&self) -> usize
```

Number of entries over all devices.

```rust
pub fn is_empty(&self) -> bool
```

Implements `Default`.

### `Scratch`

```rust
pub struct Scratch
```

A device buffer reused across calls, reallocated only when a call needs more than it holds.

**Methods**

```rust
pub fn new() -> Self
```

```rust
pub fn get<E: CubeElement>(&mut self, client: &Client, len: usize) -> Handle
```

A buffer of at least `len` elements of `E` (contents unspecified).

```rust
pub fn capacity_bytes(&self) -> usize
```

Bytes currently allocated.

Implements `Clone`, `Debug`, `Default`.

### `Workspace`

```rust
pub struct Workspace
```

Device buffers reused across the passes of an operation and across calls: buffers only grow,
nothing is freed between calls, so a multi-pass operation (separable filters, iterative
morphology, pyramids) allocates only on its first call at a given size.

A slot hands out the same buffer every time, so an array taken from a slot is valid until that
slot is taken again: use one slot per role within an operation (`"ping"`, `"pong"`, `"taps"`),
and keep a workspace per thread of work (it is not shared between concurrent calls).

**Methods**

```rust
pub fn new(client: &Client) -> Self
```

```rust
pub fn array<E: CubeElement>(
    &mut self,
    slot: &'static str,
    shape: &Shape,
) -> CoreResult<DeviceArray<E>>
```

An array of `shape` backed by `slot`'s buffer (contents unspecified). Grows the slot when it
is too small, checking the index limit and the memory budget first.

```rust
pub fn capacity_bytes(&self) -> usize
```

Bytes held by all slots.

```rust
pub fn clear(&mut self)
```

Releases every slot's buffer (CubeCL's pool keeps the memory for reuse).

```rust
pub fn client(&self) -> &Client
```

### `set_device_memory_limit`

```rust
pub fn set_device_memory_limit(bytes: Option<u64>)
```

Sets the device memory limit in bytes (`None`: no checks). Overrides `LUCEM_DEVICE_MEMORY_LIMIT`.

### `device_memory_limit`

```rust
pub fn device_memory_limit() -> Option<u64>
```

The device memory limit in bytes, if one is set.

### `memory_usage`

```rust
pub fn memory_usage(client: &Client) -> MemoryUsage
```

What CubeCL's memory pools hold on `client`'s device (all streams).

### `check_allocation`

```rust
pub fn check_allocation(client: &Client, what: &str, bytes: u64) -> CoreResult<()>
```

`Ok` if `bytes` more fit the limit next to what the device's pools already use; a no-op without a
limit or below `CHECKED_ALLOCATION_BYTES`.

### `DEVICE_MEMORY_LIMIT_ENV`

```rust
pub const DEVICE_MEMORY_LIMIT_ENV: &str = "LUCEM_DEVICE_MEMORY_LIMIT";
```

Environment variable: the device memory lucem may use, e.g. `7GiB`, `6500MB`, `7516192768`.

## `lucem_core::memory::budget`

The device memory budget: refuse an allocation that would not fit, instead of letting it fail.

### `CHECKED_ALLOCATION_BYTES`

```rust
pub const CHECKED_ALLOCATION_BYTES: u64 = _;
```

Allocations below this size are not checked: asking CubeCL for its memory report waits on the
device thread, which costs more than a small allocation is worth checking.

## `lucem_core::memory::buffer`

Typed device buffers: sizes come from the element type, never from a literal byte count.

### `bytes`

```rust
pub fn bytes<E: CubeElement>(len: usize) -> usize
```

Bytes of `len` elements of `E`, at least one element (runtimes reject empty allocations).

#### Panics
If `len` exceeds `MAX_DEVICE_ELEMENTS` (kernels index with 32-bit integers): sizes chosen by
users are checked earlier with `crate::compute::device_elements`, so this is a backstop against
a silent wrap.

### `empty`

```rust
pub fn empty<E: CubeElement>(client: &Client, len: usize) -> Handle
```

Uninitialized device buffer of `len` elements of `E`.

### `upload`

```rust
pub fn upload<E: CubeElement>(client: &Client, data: &[E]) -> Handle
```

Device copy of `data`.

### `upload_owned`

```rust
pub fn upload_owned<E: CubeElement>(client: &Client, data: Vec<E>) -> Handle
```

Device copy of `data`, taking ownership: the bytes move into the transfer without a copy.
`create` with owned bytes: `create_from_slice` copies through a slower general path (≈2–3×
longer for a 92 MB buffer on wgpu and CUDA; dsp-kitchen `tests/bench_upload.rs`).

### `write`

```rust
pub fn write<E: CubeElement>(client: &Client, handle: &Handle, data: &[E])
```

Writes `data` to the start of the existing device buffer `handle` (which holds at least
`data.len()` elements of `E`): no allocation, enqueued after the work already queued on the
client's stream, so a buffer reused every call is not overwritten before earlier kernels read it.

### `write_owned`

```rust
pub fn write_owned<E: CubeElement>(client: &Client, handle: &Handle, data: Vec<E>)
```

`fn@write` taking ownership of `data` (useful when it was filled on another thread, e.g. a
read-ahead loader).

### `write_bytes`

```rust
pub fn write_bytes(client: &Client, handle: &Handle, data: Bytes)
```

Writes `data` to the start of `handle`, narrowed to `data`'s length: a reused buffer is often
larger than the data written into it, and the CPU runtime copies the whole bound range (a write
shorter than its handle panics there; wgpu and CUDA accept it).

#### Panics
If `data` is longer than the bytes `handle` addresses.

### `zeros`

```rust
pub fn zeros<E: CubeElement + Scalar>(client: &Client, len: usize) -> Handle
```

Device buffer of `len` zeros, filled on the device (cubecl-std's zero kernel): nothing is
uploaded.

### `filled`

```rust
pub fn filled<F: ImageFloat>(client: &Client, len: usize, value: F) -> Handle
```

Device buffer of `len` copies of `value`, filled on the device: nothing is uploaded.

### `download`

```rust
pub fn download<E: CubeElement>(client: &Client, handle: Handle) -> Vec<E>
```

Host copy of a device buffer of `E` (waits for the queued work that writes it).

### `download_prefix`

```rust
pub fn download_prefix<E: CubeElement>(client: &Client, handle: Handle, len: usize) -> Vec<E>
```

Host copy of the first `len` elements of `E` of a device buffer: only those bytes are read, so a
buffer sized for the largest call costs only the current call's transfer.

### `download_range`

```rust
pub fn download_range<E: CubeElement>(
    client: &Client,
    handle: Handle,
    start: usize,
    len: usize,
) -> Vec<E>
```

Host copy of elements `start .. start + len` of `E` of a device buffer: only those bytes are read.

## `lucem_core::tiling`

Walking an image in blocks: which part of it each step reads and which part it keeps.

### `schedule`

```rust
pub fn schedule(dims: Zyx, block: Zyx, halo: Zyx) -> Vec<Block>
```

The blocks covering `dims`, in raster order (z, then y, then x), each owning `block` pixels
(less at the far edges) and reading `halo` more on each side. An axis of `block` at 0 means the
whole axis.

### `crop`

```rust
pub fn crop<T: Copy>(data: &[T], from: Zyx, at: Zyx, size: Zyx) -> Vec<T>
```

The `size` part at `at` of `data`, a C-ordered array of `from`.

#### Panics
If the part is outside `from` or `data` is shorter than `from`.
