//! The CubeCL device: the one chosen in Settings, else `LUMEN_RUNTIME` (wgpu, cuda, cpu), else the
//! first compiled in; the CPU when the chosen one does not start.

use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;

pub struct Device {
    pub client: Client,
    pub label: String,
}

impl Device {
    /// Starts the preferred backend, else the CPU one (can take a few seconds: call it off the UI
    /// thread).
    pub fn discover(preferred: Option<&str>) -> Result<Self, String> {
        let chosen = preferred.filter(|p| !p.is_empty()).and_then(|p| ComputeTarget::parse(p).ok()).filter(|t| t.is_available());
        let target = chosen.or_else(|| ComputeTarget::from_env().ok()).unwrap_or(ComputeTarget::Cpu);
        let (target, client) = match target.client() {
            Ok(c) => (target, c),
            Err(_) => (ComputeTarget::Cpu, ComputeTarget::Cpu.client().map_err(|e| format!("no compute device: {e}"))?),
        };
        Ok(Self { label: format!("{} · {}", target.name().to_uppercase(), client.name()), client })
    }
}
