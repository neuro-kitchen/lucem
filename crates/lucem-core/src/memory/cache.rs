use std::collections::BTreeMap;
use std::sync::Mutex;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{buffer, DeviceArray};
use crate::shape::Shape;

/// Constant device data (filter taps, footprint offsets, colour matrices, look-up tables) uploaded
/// once per device and key, then reused by every call.
///
/// The key always includes the device: a handle used on another device is not an error but garbage
/// (CorridorKey once reused a CUDA handle on wgpu through a cache keyed by shape alone). Keys name
/// everything that changes the data, including the element type (e.g. `"f32-sigma2-order0-r8"`).
///
/// Declare one per kind of data as a static: `static TAPS: DeviceCache = DeviceCache::new();`.
/// Entries live for the process; a cache only holds small constants.
pub struct DeviceCache {
    entries: Mutex<BTreeMap<(String, String), (Handle, Shape)>>,
}

impl DeviceCache {
    pub const fn new() -> Self {
        Self { entries: Mutex::new(BTreeMap::new()) }
    }

    /// The array stored under `key` for `client`'s device, uploading `make()` (C order, `shape`) the
    /// first time.
    ///
    /// # Panics
    /// If `make()` returns a different number of elements than `shape` holds.
    pub fn get_or_upload<E: CubeElement>(&self, client: &Client, key: &str, shape: Shape, make: impl FnOnce() -> Vec<E>) -> DeviceArray<E> {
        let entry_key = (device_key(client), format!("{}-{key}", std::any::type_name::<E>()));
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((handle, shape)) = entries.get(&entry_key) {
            return DeviceArray::from_handle(handle.clone(), shape.clone());
        }
        let data = make();
        assert_eq!(data.len(), shape.len(), "cache entry '{key}': {} values for shape {:?}", data.len(), shape.dims());
        let handle = buffer::upload_owned(client, data);
        entries.insert(entry_key, (handle.clone(), shape.clone()));
        DeviceArray::from_handle(handle, shape)
    }

    /// Drops every entry of `client`'s device (e.g. before shutting the device down).
    pub fn clear_device(&self, client: &Client) {
        let device = device_key(client);
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).retain(|(d, _), _| *d != device);
    }

    /// Number of entries over all devices.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for DeviceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The device behind `client`: the service its handles are stamped with.
fn device_key(client: &Client) -> String {
    format!("{:?}", client.service_id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_test;

    static CACHE: DeviceCache = DeviceCache::new();

    fn uploads_once_per_device_and_key(client: &Client) {
        let mut calls = 0;
        let a = CACHE.get_or_upload::<f32>(client, "test-ones", Shape::vector(4), || {
            calls += 1;
            vec![1.0; 4]
        });
        let b = CACHE.get_or_upload::<f32>(client, "test-ones", Shape::vector(4), || unreachable!("cached"));
        assert!(a.shares_buffer_with(&b) && calls == 1, "{}", client.name());
        assert_eq!(b.to_host(client), vec![1.0; 4], "{}", client.name());
        // Same key, other element type: a separate entry
        let c = CACHE.get_or_upload::<i32>(client, "test-ones", Shape::vector(4), || vec![1; 4]);
        assert_eq!(c.to_host(client), vec![1; 4], "{}", client.name());
        CACHE.clear_device(client);
    }
    runtime_test!(test_device_cache, uploads_once_per_device_and_key);
}
