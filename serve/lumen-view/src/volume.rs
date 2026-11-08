//! Volume rendering on the device: a ray per pixel through the volume, orthographic, as a maximum
//! intensity projection or front-to-back alpha compositing. One channel per launch, each with its
//! own display window; the caller blends channels in their colours.

use cubecl::prelude::*;
use lumen_core::compute::{element_position, LaunchGeometry};
use lumen_core::element::ImageFloat;
use lumen_core::memory::DeviceArray;
use lumen_core::{Shape, Zyx};

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
    let e = |err: lumen_core::CoreError| err.to_string();
    let p = DeviceArray::<f32>::from_host(client, Shape::vector(params.len()), params).map_err(e)?;
    let out = DeviceArray::<f32>::empty(client, Shape::new([height, width])).map_err(e)?;
    let geom = LaunchGeometry::elementwise(client, width * height);
    raymarch_kernel::launch::<f32>(client, geom.cube_count, geom.cube_dim, volume.binding(), p.binding(), out.binding(), width as u32, height as u32, d as u32, h as u32, w as u32, steps, if mode == VolumeMode::MaximumIntensity { 0 } else { 1 });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use lumen_core::ComputeTarget;

    use super::*;

    fn ball(n: usize) -> Vec<f32> {
        let c = n as f32 / 2.0;
        (0..n * n * n)
            .map(|i| {
                let (z, y, x) = ((i / (n * n)) as f32 + 0.5, ((i / n) % n) as f32 + 0.5, (i % n) as f32 + 0.5);
                if (z - c).powi(2) + (y - c).powi(2) + (x - c).powi(2) < (n as f32 / 5.0).powi(2) { 1.0 } else { 0.0 }
            })
            .collect()
    }

    /// Frame times, device to host included (`cargo test -p lumen-view --release -- --ignored --nocapture`).
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
