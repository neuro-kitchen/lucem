//! CubeCL GPU/CPU compute runtime discovery and management.

use lumen_core::cubecl::prelude::Client;
use lumen_core::ComputeTarget;

pub struct DeviceService {
    target: ComputeTarget,
    client: Client,
    backend_label: String,
}

impl DeviceService {
    /// Selects the preferred CubeCL backend (`LUMEN_RUNTIME` or `WGPU` -> `CUDA` -> `CPU`) and
    /// initializes its client.
    pub fn discover() -> Self {
        let target = ComputeTarget::from_env().unwrap_or(ComputeTarget::Cpu);
        let (resolved_target, client) = match target.client() {
            Ok(c) => (target, c),
            Err(_) => (
                ComputeTarget::Cpu,
                ComputeTarget::Cpu
                    .client()
                    .expect("CubeCL CPU fallback client must initialize"),
            ),
        };
        let backend_label = format!("{} · {}", resolved_target.name().to_uppercase(), client.name());
        Self {
            target: resolved_target,
            client,
            backend_label,
        }
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    pub fn backend_label(&self) -> &str {
        &self.backend_label
    }
}
