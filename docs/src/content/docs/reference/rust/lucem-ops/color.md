---
title: "lucem_ops::color"
description: "Color space conversions, Beer-Lambert histological stain unmixing, perceptual color difference metrics (CIE76, CIE94, CIEDE2000, CMC), and label map false-color overlays."
sidebar:
  label: "color"
  order: 2
---

<!-- Written by docs/scripts/rust_api.py from rustdoc's JSON: don't edit by hand. -->

Color space conversions, Beer-Lambert histological stain unmixing, perceptual color
difference metrics (CIE76, CIE94, CIEDE2000, CMC), and label map false-color overlays.

## Types

### `StainPreset`

```rust
pub enum StainPreset
```

Standard histological stain presets based on Ruifrok & Johnston (2001) optical density vectors.

**Variants**

- `HematoxylinEosinDab`: Hematoxylin + Eosin + DAB (`hed_from_rgb` / `rgb_from_hed`).
- `HematoxylinDab`: Hematoxylin + DAB + Eosin (`hdx`).
- `FeulgenLightGreen`: Feulgen + Light Green (`fgx`).
- `Giemsa`: Giemsa (Methyl Blue + Eosin).

Implements `Clone`, `Copy`, `Debug`, `Eq`, `Hash`, `PartialEq`.

## Functions

### `cie76`

```rust
pub fn cie76<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Compute pixel-wise CIE76 Euclidean color distance between two CIE L*a*b* images `[H, W, 3]`.

### `ciede94`

```rust
pub fn ciede94<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
    k_h: f64,
    k_1: f64,
    k_2: f64,
) -> OpsResult<DeviceArray<F>>
```

Compute pixel-wise CIE94 perceptual color difference between two CIE L*a*b* images `[H, W, 3]`.

### `ciede2000`

```rust
pub fn ciede2000<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
    k_h: f64,
) -> OpsResult<DeviceArray<F>>
```

Compute pixel-wise CIEDE2000 perceptual color difference between two CIE L*a*b* images `[H, W, 3]`.

### `cmc`

```rust
pub fn cmc<F: ImageFloat>(
    client: &Client,
    lab1: &DeviceArray<F>,
    lab2: &DeviceArray<F>,
    k_l: f64,
    k_c: f64,
) -> OpsResult<DeviceArray<F>>
```

Compute pixel-wise CMC(l:c) color difference between two CIE L*a*b* images `[H, W, 3]`.

### `colorize_labels`

```rust
pub fn colorize_labels<F: ImageFloat>(
    client: &Client,
    labels: &DeviceArray<u32>,
    background_gray: Option<&DeviceArray<F>>,
    bg_label: u32,
    alpha: f64,
) -> OpsResult<DeviceArray<F>>
```

Render a 2-D label map (`u32`) into an RGB overlay `[H, W, 3]`, optionally blended with a
grayscale background image `[H, W]` using opacity `alpha`.

### `rgb_to_gray`

```rust
pub fn rgb_to_gray<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert an sRGB image `[H, W, 3]` to grayscale luminance `[H, W]`.

### `gray_to_rgb`

```rust
pub fn gray_to_rgb<F: ImageFloat>(
    client: &Client,
    gray: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Replicate a grayscale image `[H, W]` into 3-channel RGB `[H, W, 3]`.

### `rgba_to_rgb`

```rust
pub fn rgba_to_rgb<F: ImageFloat>(
    client: &Client,
    rgba: &DeviceArray<F>,
    background: [f64; 3],
) -> OpsResult<DeviceArray<F>>
```

Composite an RGBA image `[H, W, 4]` onto a background RGB color `background` (default white `[1, 1, 1]`).

### `rgb_to_hsv`

```rust
pub fn rgb_to_hsv<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert RGB `[H, W, 3]` (`[0, 1]`) to HSV `[H, W, 3]` (`[0, 1]`).

### `hsv_to_rgb`

```rust
pub fn hsv_to_rgb<F: ImageFloat>(
    client: &Client,
    hsv: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert HSV `[H, W, 3]` (`[0, 1]`) to RGB `[H, W, 3]` (`[0, 1]`).

### `rgb_to_xyz`

```rust
pub fn rgb_to_xyz<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert sRGB `[H, W, 3]` to CIE 1931 XYZ (D65 illuminant) using a device-cached `3x3` matrix.

### `xyz_to_rgb`

```rust
pub fn xyz_to_rgb<F: ImageFloat>(
    client: &Client,
    xyz: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE 1931 XYZ (D65) `[H, W, 3]` to sRGB `[H, W, 3]`.

### `xyz_to_lab`

```rust
pub fn xyz_to_lab<F: ImageFloat>(
    client: &Client,
    xyz: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE XYZ (D65) `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.

### `lab_to_xyz`

```rust
pub fn lab_to_xyz<F: ImageFloat>(
    client: &Client,
    lab: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE L*a*b* `[H, W, 3]` to CIE XYZ (D65) `[H, W, 3]`.

### `rgb_to_lab`

```rust
pub fn rgb_to_lab<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert sRGB `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.

### `lab_to_rgb`

```rust
pub fn lab_to_rgb<F: ImageFloat>(
    client: &Client,
    lab: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE L*a*b* `[H, W, 3]` to sRGB `[H, W, 3]`.

### `rgb_to_luv`

```rust
pub fn rgb_to_luv<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert sRGB `[H, W, 3]` to CIE L*u*v* `[H, W, 3]`.

### `luv_to_rgb`

```rust
pub fn luv_to_rgb<F: ImageFloat>(
    client: &Client,
    luv: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE L*u*v* `[H, W, 3]` to sRGB `[H, W, 3]`.

### `lab_to_lch`

```rust
pub fn lab_to_lch<F: ImageFloat>(
    client: &Client,
    lab: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert CIE L*a*b* `[H, W, 3]` to cylindrical CIE L*C*h° `[H, W, 3]`.

### `lch_to_lab`

```rust
pub fn lch_to_lab<F: ImageFloat>(
    client: &Client,
    lch: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert cylindrical CIE L*C*h° `[H, W, 3]` to CIE L*a*b* `[H, W, 3]`.

### `rgb_to_ycbcr`

```rust
pub fn rgb_to_ycbcr<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert sRGB `[H, W, 3]` to ITU-R BT.601 YCbCr `[H, W, 3]`.

### `ycbcr_to_rgb`

```rust
pub fn ycbcr_to_rgb<F: ImageFloat>(
    client: &Client,
    ycbcr: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert ITU-R BT.601 YCbCr `[H, W, 3]` to sRGB `[H, W, 3]`.

### `rgb_to_yuv`

```rust
pub fn rgb_to_yuv<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert sRGB `[H, W, 3]` to PAL/SECAM YUV `[H, W, 3]` via a device-cached `3x3` matrix.

### `yuv_to_rgb`

```rust
pub fn yuv_to_rgb<F: ImageFloat>(
    client: &Client,
    yuv: &DeviceArray<F>,
) -> OpsResult<DeviceArray<F>>
```

Convert YUV `[H, W, 3]` to sRGB `[H, W, 3]` via a device-cached `3x3` matrix.

### `unmix_stains`

```rust
pub fn unmix_stains<F: ImageFloat>(
    client: &Client,
    rgb: &DeviceArray<F>,
    preset: StainPreset,
) -> OpsResult<DeviceArray<F>>
```

Separate an RGB image `[H, W, 3]` into optical density stain channels `[H, W, 3]` using `preset`.

### `mix_stains`

```rust
pub fn mix_stains<F: ImageFloat>(
    client: &Client,
    stains: &DeviceArray<F>,
    preset: StainPreset,
) -> OpsResult<DeviceArray<F>>
```

Recombine optical density stain channels `[H, W, 3]` back into RGB `[H, W, 3]` using `preset`.
