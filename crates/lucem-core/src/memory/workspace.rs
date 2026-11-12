use std::collections::HashMap;

use cubecl::prelude::*;

use super::{budget, buffer, DeviceArray, Scratch};
use crate::compute::device_elements;
use crate::error::CoreResult;
use crate::shape::Shape;

/// Device buffers reused across the passes of an operation and across calls: buffers only grow,
/// nothing is freed between calls, so a multi-pass operation (separable filters, iterative
/// morphology, pyramids) allocates only on its first call at a given size.
///
/// A slot hands out the same buffer every time, so an array taken from a slot is valid until that
/// slot is taken again: use one slot per role within an operation (`"ping"`, `"pong"`, `"taps"`),
/// and keep a workspace per thread of work (it is not shared between concurrent calls).
pub struct Workspace {
    client: Client,
    slots: HashMap<&'static str, Scratch>,
}

impl Workspace {
    pub fn new(client: &Client) -> Self {
        Self { client: client.clone(), slots: HashMap::new() }
    }

    /// An array of `shape` backed by `slot`'s buffer (contents unspecified). Grows the slot when it
    /// is too small, checking the index limit and the memory budget first.
    pub fn array<E: CubeElement>(&mut self, slot: &'static str, shape: &Shape) -> CoreResult<DeviceArray<E>> {
        let len = device_elements(slot, shape.dims())?;
        let scratch = self.slots.entry(slot).or_default();
        let need = buffer::bytes::<E>(len);
        if scratch.capacity_bytes() < need {
            budget::check_allocation(&self.client, slot, (need - scratch.capacity_bytes()) as u64)?;
        }
        Ok(DeviceArray::from_handle(scratch.get::<E>(&self.client, len), shape.clone()))
    }

    /// Bytes held by all slots.
    pub fn capacity_bytes(&self) -> usize {
        self.slots.values().map(Scratch::capacity_bytes).sum()
    }

    /// Releases every slot's buffer (CubeCL's pool keeps the memory for reuse).
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_test;

    fn reuses_buffers(client: &Client) {
        let mut ws = Workspace::new(client);
        let big = ws.array::<f32>("ping", &Shape::new([64, 64])).unwrap();
        let small = ws.array::<f32>("ping", &Shape::new([8, 8])).unwrap();
        assert!(big.shares_buffer_with(&small), "a smaller request reuses the slot ({})", client.name());
        let other = ws.array::<f32>("pong", &Shape::new([8, 8])).unwrap();
        assert!(!other.shares_buffer_with(&small), "slots are separate ({})", client.name());
        assert_eq!(ws.capacity_bytes(), (64 * 64 + 8 * 8) * size_of::<f32>());
    }
    runtime_test!(test_workspace_reuses_buffers, reuses_buffers);
}
