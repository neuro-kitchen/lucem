//! The CubeCL device: the one chosen in Settings, else `LUCEM_RUNTIME` (wgpu, cuda, cpu), else the
//! first compiled in; the CPU when the chosen one does not start.

use lucem_core::cubecl::prelude::Client;
use lucem_core::ComputeTarget;

pub struct Device {
    pub client: Client,
    pub label: String,
}

impl Device {
    /// Starts the preferred backend, else the CPU one (can take a few seconds: call it off the UI
    /// thread).
    pub fn discover(preferred: Option<&str>) -> Result<Self, String> {
        // A runtime without its driver is never opened (it would panic on its device thread)
        let chosen = preferred.filter(|p| !p.is_empty()).and_then(|p| ComputeTarget::parse(p).ok()).filter(|t| t.is_usable());
        let target = chosen.or_else(|| ComputeTarget::from_env().ok()).unwrap_or(ComputeTarget::Cpu);
        let (target, client) = match target.client() {
            Ok(c) => (target, c),
            Err(_) => (ComputeTarget::Cpu, ComputeTarget::Cpu.client().map_err(|e| format!("no compute device: {e}"))?),
        };
        let fallback = preferred.filter(|p| !p.is_empty()).and_then(|p| ComputeTarget::parse(p).ok()).filter(|t| *t != target).map(|t| format!(" ({} chosen: {})", t.name().to_uppercase(), t.unusable_reason().unwrap_or("not usable")));
        // Model plugins run on the same runtime: a second GPU runtime beside the app's would share
        // the GPU's memory with it and run out (the plugins have no ROCm build: their default then)
        lucem_ml::set_plugin_runtime((target != ComputeTarget::Hip).then(|| target.name()));
        // Plain words: which processor, through what
        let what = match target {
            ComputeTarget::Wgpu => "GPU · Vulkan / Metal",
            ComputeTarget::Cuda => "GPU · CUDA",
            ComputeTarget::Hip => "GPU · ROCm",
            ComputeTarget::Cpu => "CPU",
        };
        Ok(Self { label: format!("{what}{}", fallback.unwrap_or_default()), client })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings asking for a runtime whose driver is missing (CUDA without the NVIDIA module)
    /// fall back to one that runs, instead of opening it (which panics on its device thread).
    #[test]
    fn a_runtime_without_its_driver_falls_back() {
        for name in ["cuda", "hip"] {
            let t = ComputeTarget::parse(name).unwrap();
            let d = Device::discover(Some(name)).unwrap();
            if t.is_available() && !t.is_usable() {
                let opened = if name == "cuda" { "GPU · CUDA" } else { "GPU · ROCm" };
                assert!(!d.label.starts_with(opened), "{name} opened without its driver: {}", d.label);
                assert!(d.label.contains("chosen"), "the fallback is said: {}", d.label);
            }
            let a = lucem_core::memory::DeviceArray::<f32>::from_host(&d.client, lucem_core::Shape::new([4]), vec![2.0; 4]).unwrap();
            assert_eq!(a.to_host(&d.client), vec![2.0; 4], "{}", d.label);
        }
    }
}
