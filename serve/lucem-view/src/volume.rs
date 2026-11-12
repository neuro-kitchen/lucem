//! Volume rendering on the device: a ray per pixel through the volume, orthographic, as a maximum
//! intensity projection or front-to-back alpha compositing. One channel per launch, each with its
//! own display window; the caller blends channels in their colours.

use cubecl::prelude::*;
use lucem_core::compute::{element_position, LaunchGeometry};
use lucem_core::element::ImageFloat;
use lucem_core::memory::DeviceArray;
use lucem_core::{Shape, Zyx};

/// How the samples along a ray combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeMode {
    /// The brightest sample along the ray.
    MaximumIntensity,
    /// Samples composited front to back, each as opaque as it is bright.
    Composite,
}

/// Where the volume is seen from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// Radians around the vertical axis.
    pub yaw: f32,
    /// Radians up or down.
    pub pitch: f32,
    /// 1 fits the volume in the view.
    pub zoom: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { yaw: 0.6, pitch: 0.4, zoom: 1.0 }
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn raymarch_kernel<F: ImageFloat>(volume: &[F], p: &[F], out: &mut [F], width: u32, height: u32, nz: u32, ny: u32, nx: u32, steps: u32, mode: u32) {
    let idx = element_position() as u32;
    if idx < width * height {
        let col = idx % width;
        let row = idx / width;
        let half = F::new(0.5f32);
        let u = (F::cast_from(col) + half - F::cast_from(width) * half) * p[9usize];
        let v = (F::cast_from(height) * half - F::cast_from(row) - half) * p[9usize];
        let back = p[10usize];
        let (hx, hy, hz) = (p[11usize], p[12usize], p[13usize]);
        let (dx, dy, dz) = (p[6usize], p[7usize], p[8usize]);
        let ox = p[0usize] * u + p[3usize] * v - dx * back;
        let oy = p[1usize] * u + p[4usize] * v - dy * back;
        let oz = p[2usize] * u + p[5usize] * v - dz * back;
        // Where the ray enters and leaves the box (the camera keeps the direction's parts non-zero)
        let (t1, t2) = ((-hx - ox) / dx, (hx - ox) / dx);
        let (t3, t4) = ((-hy - oy) / dy, (hy - oy) / dy);
        let (t5, t6) = ((-hz - oz) / dz, (hz - oz) / dz);
        let near = F::max(F::max(F::min(t1, t2), F::min(t3, t4)), F::min(t5, t6));
        let far = F::min(F::min(F::max(t1, t2), F::max(t3, t4)), F::max(t5, t6));
        let mut acc = F::new(0.0f32);
        if far > near {
            let dt = (far - near) / F::cast_from(steps);
            let mut opacity = F::new(0.0f32);
            let (lo, inv) = (p[14usize], p[15usize]);
            let (fx, fy, fz) = (F::cast_from(nx), F::cast_from(ny), F::cast_from(nz));
            let one = F::new(1.0f32);
            for i in 0..steps {
                let t = near + (F::cast_from(i) + half) * dt;
                let cx = F::max(F::min(F::floor((ox + dx * t + hx) / (hx + hx) * fx), fx - one), F::new(0.0f32));
                let cy = F::max(F::min(F::floor((oy + dy * t + hy) / (hy + hy) * fy), fy - one), F::new(0.0f32));
                let cz = F::max(F::min(F::floor((oz + dz * t + hz) / (hz + hz) * fz), fz - one), F::new(0.0f32));
                let at = (u32::cast_from(cz) * ny + u32::cast_from(cy)) * nx + u32::cast_from(cx);
                let s = F::max(F::min((volume[at as usize] - lo) * inv, one), F::new(0.0f32));
                if mode == 0u32 {
                    acc = F::max(acc, s);
                } else {
                    let a = F::min(s * p[16usize], one);
                    acc += (one - opacity) * a;
                    opacity += (one - opacity) * a;
                }
            }
        }
        out[idx as usize] = acc;
    }
}

/// Renders `volume` (`dims`, `[d, h, w]`; `spacing`, the size of a voxel along z, y, x) into a
/// `width × height` image with values 0–1 after `window`.
#[allow(clippy::too_many_arguments)]
pub fn render_volume(client: &Client, volume: &DeviceArray<f32>, dims: Zyx, spacing: [f32; 3], camera: OrbitCamera, (width, height): (usize, usize), window: (f32, f32), mode: VolumeMode) -> Result<DeviceArray<f32>, String> {
    let [d, h, w] = dims;
    if volume.len() != d * h * w || width == 0 || height == 0 {
        return Err("the volume or the image is empty or of the wrong size".into());
    }
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    let nonzero = |v: f32| if v.abs() < 1e-4 { 1e-4 } else { v };
    let forward = [nonzero(sy * cp), nonzero(sp), nonzero(cy * cp)];
    let right = [cy, 0.0, -sy];
    let up = [-sy * sp, cp, -cy * sp];
    let extent = [w as f32 * spacing[2], h as f32 * spacing[1], d as f32 * spacing[0]];
    let largest = extent[0].max(extent[1]).max(extent[2]).max(1e-6);
    let half = [extent[0] / largest, extent[1] / largest, extent[2] / largest];
    let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    let scale = 2.0 * radius / (width.min(height) as f32 * camera.zoom.max(1e-3));
    let span = (window.1 - window.0).abs().max(1e-12);
    let steps = (2 * d.max(h).max(w)).clamp(16, 1024) as u32;
    let alpha = (32.0 / steps as f32).min(1.0);
    #[rustfmt::skip]
    let params = vec![
        right[0], right[1], right[2], up[0], up[1], up[2], forward[0], forward[1], forward[2],
        scale, 2.0 * radius, half[0], half[1], half[2], window.0, 1.0 / span, alpha,
    ];
    let e = |err: lucem_core::CoreError| err.to_string();
    let p = DeviceArray::<f32>::from_host(client, Shape::vector(params.len()), params).map_err(e)?;
    let out = DeviceArray::<f32>::empty(client, Shape::new([height, width])).map_err(e)?;
    let geom = LaunchGeometry::elementwise(client, width * height);
    raymarch_kernel::launch::<f32>(client, geom.cube_count, geom.cube_dim, volume.binding(), p.binding(), out.binding(), width as u32, height as u32, d as u32, h as u32, w as u32, steps, if mode == VolumeMode::MaximumIntensity { 0 } else { 1 });
    Ok(out)
}

/// One channel of [`render_volume_rgba`]: shown or not, its display window, its colour (linear
/// RGB, 0–1) and its gamma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeChannel {
    pub visible: bool,
    pub window: (f32, f32),
    pub color: [f32; 3],
    pub gamma: f32,
}

/// How finely [`render_volume_rgba`] samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeQuality {
    /// Nearest voxel, a step of 1.5 voxels: while the view turns.
    Fast,
    /// Trilinear, a step of 0.75 voxel: at rest.
    Fine,
}

/// Floats per channel in the channel table of the kernel.
const CHANNEL_PARAMS: usize = 7;

/// A value of `volume` (channel at `base`) at the continuous voxel position `(gx, gy, gz)` (voxel
/// centres at integers), nearest (`smooth == 0`) or trilinear. Positions are clamped to the box.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn sample_voxel<F: ImageFloat>(volume: &[F], base: u32, gx: F, gy: F, gz: F, nx: u32, ny: u32, nz: u32, smooth: u32) -> F {
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let half = F::new(0.5f32);
    let x = F::max(F::min(gx, F::cast_from(nx) - one), zero);
    let y = F::max(F::min(gy, F::cast_from(ny) - one), zero);
    let z = F::max(F::min(gz, F::cast_from(nz) - one), zero);
    let mut out = zero;
    if smooth == 0u32 {
        let xi = u32::min(u32::cast_from(F::floor(x + half)), nx - 1u32);
        let yi = u32::min(u32::cast_from(F::floor(y + half)), ny - 1u32);
        let zi = u32::min(u32::cast_from(F::floor(z + half)), nz - 1u32);
        out = volume[(base + (zi * ny + yi) * nx + xi) as usize];
    } else {
        let (xf, yf, zf) = (F::floor(x), F::floor(y), F::floor(z));
        let (tx, ty, tz) = (x - xf, y - yf, z - zf);
        let x0 = u32::cast_from(xf);
        let y0 = u32::cast_from(yf);
        let z0 = u32::cast_from(zf);
        let x1 = u32::min(x0 + 1u32, nx - 1u32);
        let y1 = u32::min(y0 + 1u32, ny - 1u32);
        let z1 = u32::min(z0 + 1u32, nz - 1u32);
        let r00 = base + (z0 * ny + y0) * nx;
        let r01 = base + (z0 * ny + y1) * nx;
        let r10 = base + (z1 * ny + y0) * nx;
        let r11 = base + (z1 * ny + y1) * nx;
        let a00 = volume[(r00 + x0) as usize] + (volume[(r00 + x1) as usize] - volume[(r00 + x0) as usize]) * tx;
        let a01 = volume[(r01 + x0) as usize] + (volume[(r01 + x1) as usize] - volume[(r01 + x0) as usize]) * tx;
        let a10 = volume[(r10 + x0) as usize] + (volume[(r10 + x1) as usize] - volume[(r10 + x0) as usize]) * tx;
        let a11 = volume[(r11 + x0) as usize] + (volume[(r11 + x1) as usize] - volume[(r11 + x0) as usize]) * tx;
        let b0 = a00 + (a01 - a00) * ty;
        let b1 = a10 + (a11 - a10) * ty;
        out = b0 + (b1 - b0) * tz;
    }
    out
}

/// All channels of `volume` (`n_volume` stacked `[c, z, y, x]`) and `extra` (`n_extra`, the
/// same layout) along one ray per pixel, blended in their colours into packed BGRA (`b | g << 8 |
/// r << 16 | a << 24`). `p` is the camera table of [`render_volume`] plus the step length at 17;
/// `ch` holds [`CHANNEL_PARAMS`] floats per channel: shown, low, 1 / span, r, g, b, 1 / gamma.
#[cube(launch)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub(crate) fn raymarch_rgba_kernel<F: ImageFloat>(
    volume: &[F],
    extra: &[F],
    p: &[F],
    ch: &[F],
    out: &mut [u32],
    width: u32,
    height: u32,
    nz: u32,
    ny: u32,
    nx: u32,
    n_volume: u32,
    n_extra: u32,
    max_steps: u32,
    mode: u32,
    smooth: u32,
) {
    let idx = element_position() as u32;
    if idx < width * height {
        let col = idx % width;
        let row = idx / width;
        let half = F::new(0.5f32);
        let zero = F::new(0.0f32);
        let one = F::new(1.0f32);
        let u = (F::cast_from(col) + half - F::cast_from(width) * half) * p[9usize];
        let v = (F::cast_from(height) * half - F::cast_from(row) - half) * p[9usize];
        let back = p[10usize];
        let (hx, hy, hz) = (p[11usize], p[12usize], p[13usize]);
        let (dx, dy, dz) = (p[6usize], p[7usize], p[8usize]);
        let ox = p[0usize] * u + p[3usize] * v - dx * back;
        let oy = p[1usize] * u + p[4usize] * v - dy * back;
        let oz = p[2usize] * u + p[5usize] * v - dz * back;
        let (t1, t2) = ((-hx - ox) / dx, (hx - ox) / dx);
        let (t3, t4) = ((-hy - oy) / dy, (hy - oy) / dy);
        let (t5, t6) = ((-hz - oz) / dz, (hz - oz) / dz);
        let near = F::max(F::max(F::min(t1, t2), F::min(t3, t4)), F::min(t5, t6));
        let far = F::min(F::min(F::max(t1, t2), F::max(t3, t4)), F::max(t5, t6));
        let mut red = zero;
        let mut green = zero;
        let mut blue = zero;
        if far > near {
            let step = p[17usize];
            let mut n = u32::cast_from(F::floor((far - near) / step)) + 1u32;
            if n > max_steps {
                n = max_steps;
            }
            let dt = (far - near) / F::cast_from(n);
            let (fx, fy, fz) = (F::cast_from(nx), F::cast_from(ny), F::cast_from(nz));
            let alpha = p[16usize];
            let voxels = nz * ny * nx;
            let channels = n_volume + n_extra;
            for c in 0..channels {
                let k = (c * 7u32) as usize;
                if ch[k] > half {
                    let lo = ch[k + 1usize];
                    let inv = ch[k + 2usize];
                    let mut acc = zero;
                    let mut opacity = zero;
                    let mut i = 0u32;
                    while i < n {
                        let t = near + (F::cast_from(i) + half) * dt;
                        let gx = (ox + dx * t + hx) / (hx + hx) * fx - half;
                        let gy = (oy + dy * t + hy) / (hy + hy) * fy - half;
                        let gz = (oz + dz * t + hz) / (hz + hz) * fz - half;
                        let mut raw = zero;
                        if c < n_volume {
                            raw = sample_voxel::<F>(volume, c * voxels, gx, gy, gz, nx, ny, nz, smooth);
                        } else {
                            raw = sample_voxel::<F>(extra, (c - n_volume) * voxels, gx, gy, gz, nx, ny, nz, smooth);
                        }
                        let s = F::max(F::min((raw - lo) * inv, one), zero);
                        if mode == 0u32 {
                            acc = F::max(acc, s);
                        } else {
                            let a = F::min(s * alpha, one);
                            acc += (one - opacity) * a;
                            opacity += (one - opacity) * a;
                            if opacity > F::new(0.98f32) {
                                break;
                            }
                        }
                        i += 1u32;
                    }
                    if acc > zero {
                        let shade = F::powf(acc, ch[k + 6usize]);
                        red += ch[k + 3usize] * shade;
                        green += ch[k + 4usize] * shade;
                        blue += ch[k + 5usize] * shade;
                    }
                }
            }
        }
        let scale = F::new(255.0f32);
        let r8 = u32::cast_from(F::min(red, one) * scale + half);
        let g8 = u32::cast_from(F::min(green, one) * scale + half);
        let b8 = u32::cast_from(F::min(blue, one) * scale + half);
        out[idx as usize] = b8 | (g8 << 8u32) | (r8 << 16u32) | (255u32 << 24u32);
    }
}

/// The camera of a render: the unit vectors and the box half extents, as the kernels use them.
struct CameraSetup {
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    half: [f32; 3],
    radius: f32,
    /// World units per image pixel.
    scale: f32,
}

fn camera_setup(dims: Zyx, spacing: [f32; 3], camera: OrbitCamera, (width, height): (usize, usize)) -> CameraSetup {
    let [d, h, w] = dims;
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    let nonzero = |v: f32| if v.abs() < 1e-4 { 1e-4 } else { v };
    let forward = [nonzero(sy * cp), nonzero(sp), nonzero(cy * cp)];
    let right = [cy, 0.0, -sy];
    let up = [-sy * sp, cp, -cy * sp];
    let extent = [w as f32 * spacing[2], h as f32 * spacing[1], d as f32 * spacing[0]];
    let largest = extent[0].max(extent[1]).max(extent[2]).max(1e-6);
    let half = [extent[0] / largest, extent[1] / largest, extent[2] / largest];
    let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    let scale = 2.0 * radius / (width.min(height).max(1) as f32 * camera.zoom.max(1e-3));
    CameraSetup { right, up, forward, half, radius, scale }
}

/// Where a point of the volume lands in a `width × height` render: `frac` is `[z, y, x]` as
/// fractions of the box (0 and 1 are its faces); the result is in image pixels.
pub fn project_point(dims: Zyx, spacing: [f32; 3], camera: OrbitCamera, (width, height): (usize, usize), frac: [f32; 3]) -> [f32; 2] {
    let c = camera_setup(dims, spacing, camera, (width, height));
    let world = [(2.0 * frac[2] - 1.0) * c.half[0], (2.0 * frac[1] - 1.0) * c.half[1], (2.0 * frac[0] - 1.0) * c.half[2]];
    let dot = |a: [f32; 3]| a[0] * world[0] + a[1] * world[1] + a[2] * world[2];
    let (u, v) = (dot(c.right) / c.scale, dot(c.up) / c.scale);
    [u + width as f32 / 2.0, height as f32 / 2.0 - v]
}

/// Renders every channel of `volume` (`[c, d, h, w]`, `dims = [d, h, w]`) and of `extra` (label
/// channels of the same size, or `None`) into one `width × height` image of packed BGRA pixels
/// (one `u32` each, bytes in BGRA order): one launch and one read back for all channels, blended
/// in their colours on the device. `channels` describes the channels of `volume`, then those of
/// `extra`.
#[allow(clippy::too_many_arguments)]
pub fn render_volume_rgba(
    client: &Client,
    volume: &DeviceArray<f32>,
    extra: Option<&DeviceArray<f32>>,
    dims: Zyx,
    spacing: [f32; 3],
    camera: OrbitCamera,
    (width, height): (usize, usize),
    channels: &[VolumeChannel],
    mode: VolumeMode,
    quality: VolumeQuality,
) -> Result<DeviceArray<u32>, String> {
    let [d, h, w] = dims;
    let voxels = d * h * w;
    if voxels == 0 || width == 0 || height == 0 || !volume.len().is_multiple_of(voxels) {
        return Err("the volume or the image is empty or of the wrong size".into());
    }
    let n_volume = volume.len() / voxels;
    let n_extra = extra.map_or(0, |e| e.len() / voxels);
    if channels.len() != n_volume + n_extra {
        return Err(format!("{} channel settings for {} channels", channels.len(), n_volume + n_extra));
    }
    let c = camera_setup(dims, spacing, camera, (width, height));
    // One step per voxel along the finest axis (0.75 at rest, 1.5 while turning)
    let voxel = (0..3).map(|a| 2.0 * c.half[a] / [w, h, d][a].max(1) as f32).fold(f32::MAX, f32::min);
    let step = voxel * if quality == VolumeQuality::Fine { 0.75 } else { 1.5 };
    // Opacity per unit length, so the look does not change with the step
    let alpha = (step * 12.0).min(1.0);
    #[rustfmt::skip]
    let params = vec![
        c.right[0], c.right[1], c.right[2], c.up[0], c.up[1], c.up[2], c.forward[0], c.forward[1], c.forward[2],
        c.scale, 2.0 * c.radius, c.half[0], c.half[1], c.half[2], 0.0, 0.0, alpha, step,
    ];
    let mut table = Vec::with_capacity(channels.len() * CHANNEL_PARAMS);
    for ch in channels {
        let span = (ch.window.1 - ch.window.0).abs().max(1e-12);
        table.extend_from_slice(&[if ch.visible { 1.0 } else { 0.0 }, ch.window.0, 1.0 / span, ch.color[0], ch.color[1], ch.color[2], 1.0 / ch.gamma.max(0.05)]);
    }
    let e = |err: lucem_core::CoreError| err.to_string();
    let p = DeviceArray::<f32>::from_host(client, Shape::vector(params.len()), params).map_err(e)?;
    let t = DeviceArray::<f32>::from_host(client, Shape::vector(table.len()), table).map_err(e)?;
    let dummy;
    let extra = match extra {
        Some(x) => x,
        None => {
            dummy = DeviceArray::<f32>::from_host(client, Shape::vector(1), vec![0.0]).map_err(e)?;
            &dummy
        }
    };
    let out = DeviceArray::<u32>::empty(client, Shape::new([height, width])).map_err(e)?;
    let max_steps = (4 * d.max(h).max(w)).clamp(16, 4096) as u32;
    let geom = LaunchGeometry::elementwise(client, width * height);
    raymarch_rgba_kernel::launch::<f32>(
        client,
        geom.cube_count,
        geom.cube_dim,
        volume.binding(),
        extra.binding(),
        p.binding(),
        t.binding(),
        out.binding(),
        width as u32,
        height as u32,
        d as u32,
        h as u32,
        w as u32,
        n_volume as u32,
        n_extra as u32,
        max_steps,
        if mode == VolumeMode::MaximumIntensity { 0 } else { 1 },
        if quality == VolumeQuality::Fine { 1 } else { 0 },
    );
    Ok(out)
}

/// Packed BGRA pixels (as [`render_volume_rgba`] returns them) as bytes in BGRA order.
pub fn unpack_bgra(pixels: &[u32]) -> Vec<u8> {
    pixels.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use lucem_core::ComputeTarget;

    use super::*;

    #[test]
    fn two_channels_render_in_one_launch_in_their_colours() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            let n = 32;
            let mut both = ball(n);
            both.extend(vec![0.0; n * n * n]);
            let v = DeviceArray::from_host(&client, Shape::new([2 * n, n, n]), both).unwrap();
            let red = VolumeChannel { visible: true, window: (0.0, 1.0), color: [1.0, 0.0, 0.0], gamma: 1.0 };
            let blue = VolumeChannel { color: [0.0, 0.0, 1.0], ..red };
            for quality in [VolumeQuality::Fast, VolumeQuality::Fine] {
                let px = render_volume_rgba(&client, &v, None, [n, n, n], [1.0; 3], OrbitCamera::default(), (48, 48), &[red, blue], VolumeMode::MaximumIntensity, quality).unwrap().to_host(&client);
                let bytes = unpack_bgra(&px);
                let centre = (24 * 48 + 24) * 4;
                assert_eq!(bytes[centre + 3], 255, "{target}: opaque");
                assert!(bytes[centre + 2] > 200, "{target} {quality:?}: red at the centre, {:?}", &bytes[centre..centre + 4]);
                assert_eq!(bytes[centre], 0, "{target}: no blue (the second channel is empty)");
                assert_eq!(&bytes[0..3], &[0, 0, 0], "{target}: black corner");
            }
        }
    }

    /// Frame times of the one-launch path, two channels, read back included
    /// (`cargo test -p lucem-view -- --ignored --nocapture rgba_frame_times`).
    #[test]
    #[ignore = "a timing, for a machine with a GPU"]
    fn rgba_frame_times() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            for (n, image, quality) in [(96usize, 320usize, VolumeQuality::Fast), (256, 512, VolumeQuality::Fine)] {
                let mut both = ball(n);
                both.extend(ball(n));
                let v = DeviceArray::from_host(&client, Shape::new([2 * n, n, n]), both).unwrap();
                let ch = VolumeChannel { visible: true, window: (0.0, 1.0), color: [1.0, 0.5, 0.2], gamma: 1.0 };
                for mode in [VolumeMode::MaximumIntensity, VolumeMode::Composite] {
                    let mut times = Vec::new();
                    for i in 0..12 {
                        let t0 = std::time::Instant::now();
                        let camera = OrbitCamera { yaw: i as f32 * 0.1, ..OrbitCamera::default() };
                        let px = render_volume_rgba(&client, &v, None, [n, n, n], [1.0; 3], camera, (image, image), &[ch, ch], mode, quality).unwrap().to_host(&client);
                        assert_eq!(px.len(), image * image);
                        times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    times.sort_by(f64::total_cmp);
                    println!("{target}: 2 × {n}³ voxels to {image}² px, {mode:?} {quality:?}: median {:.1} ms, worst {:.1} ms", times[6], times[11]);
                }
            }
        }
    }

    #[test]
    fn the_box_centre_projects_to_the_image_centre() {
        let p = project_point([10, 20, 30], [1.0; 3], OrbitCamera::default(), (100, 80), [0.5, 0.5, 0.5]);
        assert!((p[0] - 50.0).abs() < 1e-3 && (p[1] - 40.0).abs() < 1e-3, "{p:?}");
    }

    fn ball(n: usize) -> Vec<f32> {
        let c = n as f32 / 2.0;
        (0..n * n * n)
            .map(|i| {
                let (z, y, x) = ((i / (n * n)) as f32 + 0.5, ((i / n) % n) as f32 + 0.5, (i % n) as f32 + 0.5);
                if (z - c).powi(2) + (y - c).powi(2) + (x - c).powi(2) < (n as f32 / 5.0).powi(2) { 1.0 } else { 0.0 }
            })
            .collect()
    }

    /// Frame times, device to host included (`cargo test -p lucem-view --release -- --ignored --nocapture`).
    #[test]
    #[ignore = "a timing, for a machine with a GPU"]
    fn frame_times() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            for (n, image) in [(96usize, 320usize), (256, 1024)] {
                let v = DeviceArray::from_host(&client, Shape::new([n, n, n]), ball(n)).unwrap();
                let mut times = Vec::new();
                for i in 0..12 {
                    let t0 = std::time::Instant::now();
                    let camera = OrbitCamera { yaw: i as f32 * 0.1, ..OrbitCamera::default() };
                    let host = render_volume(&client, &v, [n, n, n], [1.0; 3], camera, (image, image), (0.0, 1.0), VolumeMode::Composite).unwrap().to_host(&client);
                    assert_eq!(host.len(), image * image);
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }
                times.sort_by(f64::total_cmp);
                println!("{target}: {n}³ voxels to {image}² px: median {:.1} ms, worst {:.1} ms", times[6], times[11]);
            }
        }
    }

    #[test]
    fn a_ball_is_bright_at_the_centre_and_dark_at_the_corners() {
        for target in ComputeTarget::for_tests() {
            let client = target.client().unwrap();
            let n = 48;
            let v = DeviceArray::from_host(&client, Shape::new([n, n, n]), ball(n)).unwrap();
            for mode in [VolumeMode::MaximumIntensity, VolumeMode::Composite] {
                let img = render_volume(&client, &v, [n, n, n], [1.0; 3], OrbitCamera::default(), (64, 64), (0.0, 1.0), mode).unwrap().to_host(&client);
                assert!(img[32 * 64 + 32] > 0.5, "{target} {mode:?}: centre {}", img[32 * 64 + 32]);
                assert_eq!(img[0], 0.0, "{target} {mode:?}: corner");
            }
        }
    }
}
