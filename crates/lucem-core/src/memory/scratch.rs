use cubecl::prelude::*;
use cubecl::server::Handle;

use super::buffer;

/// A device buffer reused across calls, reallocated only when a call needs more than it holds.
#[derive(Debug, Clone, Default)]
pub struct Scratch {
    handle: Option<Handle>,
    capacity_bytes: usize,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// A buffer of at least `len` elements of `E` (contents unspecified), rounded to a size class
    /// (`2^k × {1, 1.25, 1.5, 1.75}`) so CubeCL's pool reuses freed pages across nearby sizes.
    pub fn get<E: CubeElement>(&mut self, client: &Client, len: usize) -> Handle {
        let need = super::budget::size_class(buffer::bytes::<E>(len));
        match &self.handle {
            Some(handle) if self.capacity_bytes >= need => handle.clone(),
            _ => {
                let handle = client.empty(need);
                self.handle = Some(handle.clone());
                self.capacity_bytes = need;
                handle
            }
        }
    }

    /// Bytes currently allocated.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }
}
