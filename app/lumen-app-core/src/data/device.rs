//! The CubeCL device: `LUMEN_RUNTIME` (wgpu, cuda, cpu) or the first that starts.

use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;

pub struct Device {
    pub client: Client,
    pub label: String,
}

impl Device {
    /// Starts the preferred backend, else the CPU one (can take a few seconds: call it off the UI
    /// thread).
    pub fn discover() -> Result<Self, String> {
        let target = ComputeTarget::from_env().unwrap_or(ComputeTarget::Cpu);
        let (target, client) = match target.client() {
            Ok(c) => (target, c),
            Err(_) => (ComputeTarget::Cpu, ComputeTarget::Cpu.client().map_err(|e| format!("no compute device: {e}"))?),
        };
        Ok(Self { label: format!("{} · {}", target.name().to_uppercase(), client.name()), client })
    }
}
