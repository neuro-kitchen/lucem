use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{budget, buffer};
use crate::compute::device_elements;
use crate::element::ImageFloat;
use crate::error::{CoreError, CoreResult};
use crate::shape::Shape;

/// A C-contiguous array of `E` on one device: what every lucem operation takes and returns.
///
/// Data moves between host and device only through [`Self::from_host`] / [`Self::write_from_host`]
/// and [`Self::to_host`] / [`Self::to_host_range`]; everything in between stays on the device.
///
/// Cloning is cheap and **shares the buffer** (as Burn tensors do): never write into an array you
/// did not allocate or take from a [`Workspace`](super::Workspace). The handle may address more
/// bytes than the array uses (a reused buffer); [`Self::binding`] covers exactly `len` elements.
#[derive(Debug, Clone)]
pub struct DeviceArray<E: CubeElement> {
    handle: Handle,
    shape: Shape,
    _elem: PhantomData<E>,
}

impl<E: CubeElement> DeviceArray<E> {
    /// Uninitialized array. Checks the 32-bit index limit and, for large arrays, the memory budget.
    pub fn empty(client: &Client, shape: Shape) -> CoreResult<Self> {
        Self::reserve(client, "array", &shape)?;
        Ok(Self::from_handle(buffer::empty::<E>(client, shape.len()), shape))
    }

    /// Device copy of `data` (C order), moving the bytes into the transfer (the fast upload path).
    pub fn from_host(client: &Client, shape: Shape, data: Vec<E>) -> CoreResult<Self> {
        Self::expect_len(&shape, data.len())?;
        Self::reserve(client, "array", &shape)?;
        Ok(Self::from_handle(buffer::upload_owned(client, data), shape))
    }

    /// Device copy of `data` (copies the slice once on the host, then as [`Self::from_host`]).
    pub fn from_slice(client: &Client, shape: Shape, data: &[E]) -> CoreResult<Self> {
        Self::from_host(client, shape, data.to_vec())
    }

    /// An array over an existing buffer that holds at least `shape.len()` elements of `E` (from a
    /// [`Scratch`](super::Scratch), [`Workspace`](super::Workspace) or [`DeviceCache`](super::DeviceCache)).
    ///
    /// # Panics
    /// If the buffer is smaller than the shape needs.
    pub fn from_handle(handle: Handle, shape: Shape) -> Self {
        let need = buffer::bytes::<E>(shape.len()) as u64;
        assert!(handle.size_in_used() >= need, "buffer of {} bytes is too small for {:?} ({need} bytes)", handle.size_in_used(), shape.dims());
        Self { handle, shape, _elem: PhantomData }
    }

    /// Writes `data` (C order, exactly `len` elements) into this array's buffer: no allocation,
    /// queued after the work already on the stream (an array refilled for every image of a batch).
    pub fn write_from_host(&self, client: &Client, data: Vec<E>) -> CoreResult<()> {
        Self::expect_len(&self.shape, data.len())?;
        buffer::write_owned(client, &self.handle, data);
        Ok(())
    }

    /// Host copy of the whole array (waits for the queued work that writes it).
    pub fn to_host(&self, client: &Client) -> Vec<E> {
        buffer::download_prefix(client, self.handle.clone(), self.len())
    }

    /// Host copy of elements `start .. start + len` (C order): only those bytes are transferred.
    ///
    /// # Panics
    /// If the range exceeds the array.
    pub fn to_host_range(&self, client: &Client, start: usize, len: usize) -> Vec<E> {
        assert!(start + len <= self.len(), "elements {start}..{} exceed the array's {}", start + len, self.len());
        buffer::download_range(client, self.handle.clone(), start, len)
    }

    /// The kernel argument for this array: a binding of exactly `len` elements.
    pub fn binding(&self) -> BufferArg {
        // SAFETY: `from_handle` checked that the buffer holds `len` elements of `E`
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len()) }
    }

    /// Same buffer, other shape with the same number of elements (a free reshape).
    pub fn reshaped(&self, shape: Shape) -> CoreResult<Self> {
        Self::expect_len(&shape, self.len())?;
        Ok(Self { handle: self.handle.clone(), shape, _elem: PhantomData })
    }

    /// Whether `self` and `other` use the same device buffer (kernels must not bind both: no aliased
    /// bindings). Clones of a handle share one memory descriptor, so pointer identity decides.
    pub fn shares_buffer_with<T: CubeElement>(&self, other: &DeviceArray<T>) -> bool {
        self.handle.service == other.handle.service
            && std::ptr::eq(self.handle.memory.descriptor(), other.handle.memory.descriptor())
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn len(&self) -> usize {
        self.shape.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    pub fn into_handle(self) -> Handle {
        self.handle
    }

    /// Checks a new array of `shape`: the 32-bit index limit, then the memory budget.
    fn reserve(client: &Client, what: &str, shape: &Shape) -> CoreResult<()> {
        let len = device_elements(what, shape.dims())?;
        budget::check_allocation(client, what, (len.max(1) * size_of::<E>()) as u64)
    }

    fn expect_len(shape: &Shape, len: usize) -> CoreResult<()> {
        if shape.len() == len {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch { expected: shape.dims().to_vec(), actual: vec![len] })
        }
    }
}

impl<F: ImageFloat> DeviceArray<F> {
    /// Array of `value`, filled on the device (nothing uploaded).
    pub fn filled(client: &Client, shape: Shape, value: F) -> CoreResult<Self> {
        Self::reserve(client, "array", &shape)?;
        Ok(Self::from_handle(buffer::filled(client, shape.len(), value), shape))
    }
}

impl<E: CubeElement + Scalar> DeviceArray<E> {
    /// Array of zeros, filled on the device (nothing uploaded).
    pub fn zeros(client: &Client, shape: Shape) -> CoreResult<Self> {
        Self::reserve(client, "array", &shape)?;
        Ok(Self::from_handle(buffer::zeros::<E>(client, shape.len()), shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_test;

    fn round_trips(client: &Client) {
        let shape = Shape::new([3, 5, 7]);
        let data: Vec<f32> = (0..shape.len()).map(|i| i as f32 * 0.5 - 20.0).collect();
        let a = DeviceArray::from_host(client, shape.clone(), data.clone()).unwrap();
        assert_eq!(a.to_host(client), data, "{}", client.name());
        assert_eq!(a.to_host_range(client, 10, 4), data[10..14].to_vec(), "{}", client.name());

        let b = a.reshaped(Shape::new([15, 7])).unwrap();
        assert_eq!(b.to_host(client), data, "{}", client.name());
        assert!(a.reshaped(Shape::new([2, 7])).is_err());

        a.write_from_host(client, vec![1.0; shape.len()]).unwrap();
        assert!(b.to_host(client).iter().all(|&x| x == 1.0), "clones share the buffer ({})", client.name());
        assert!(DeviceArray::from_host(client, shape, vec![0.0f32; 3]).is_err(), "wrong length is refused");

        let z = DeviceArray::<f32>::zeros(client, Shape::new([1_000_003])).unwrap();
        assert!(z.to_host(client).iter().all(|&x| x == 0.0), "{}", client.name());
    }
    runtime_test!(test_device_array_round_trips, round_trips);
}
