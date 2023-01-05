use std::{marker::PhantomData, mem, sync::Arc};

use bytemuck::Pod;
use wgpu::util::DeviceExt as _;

use crate::Context;

/// Specifies the storage access of the buffer in the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferAccess {
    /// The buffer can only be read.
    ///
    /// Corresponds to a `var<storage, read>` in wgsl.
    ReadOnly,

    /// The buffer may be read and/or written.
    ///
    /// Corresponds to a `var<storage, read_write>` in wgsl.
    ReadWrite,
}

/// Buffer on the GPU that stores homogeneous data.
///
/// With multiple elements it acts as an `array<T>` in kernel code.
/// With a single element it can also act as something of type `T`.
#[derive(Debug)]
pub struct Buffer<T> {
    pub(crate) device: Arc<crate::Device>,
    pub(crate) handle: wgpu::Buffer,
    _marker: PhantomData<Vec<T>>,
}

impl<T: Pod> Buffer<T> {
    // cheesy workaround to be able to make a const bitflag
    // see: https://github.com/bitflags/bitflags/issues/180
    const USAGES: wgpu::BufferUsages = wgpu::BufferUsages::from_bits_truncate(
        wgpu::BufferUsages::STORAGE.bits()
            | wgpu::BufferUsages::COPY_DST.bits()
            | wgpu::BufferUsages::COPY_SRC.bits(),
    );

    /// Allocate a buffer on the GPU with `capacity` **elements of T**.
    ///
    /// # Panics
    ///
    /// - if capacity exceeds the limit of `max_buffer_size` (with a default
    /// value of **2^30 bytes** that can be configured in `ContextInfo`).
    pub fn new(context: &Context, capacity: wgpu::BufferAddress) -> Self {
        let buffer = context
            .device
            .handle
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("buffer"),
                size: capacity * mem::size_of::<T>() as wgpu::BufferAddress,
                usage: Self::USAGES,
                mapped_at_creation: false,
            });

        Self {
            device: Arc::clone(&context.device),
            handle: buffer,
            _marker: PhantomData,
        }
    }

    /// Creates an empty buffer able to store the same ammount of data that `original` does.
    pub fn empty_like(original: &Self) -> Self {
        let buffer = original
            .device
            .handle
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("buffer"),
                size: original.handle.size(),
                usage: Self::USAGES,
                mapped_at_creation: false,
            });

        Self {
            device: Arc::clone(&original.device),
            handle: buffer,
            _marker: PhantomData,
        }
    }

    /// Write to a buffer starting at `index`.
    ///
    /// # Panics
    ///
    /// - if `data` overruns the buffer from any index.
    pub fn write(&self, data: &[T], index: wgpu::BufferAddress) {
        let offset = index * mem::size_of::<T>() as u64;
        self.device
            .queue
            .write_buffer(&self.handle, offset, bytemuck::cast_slice(data));
    }

    /// Allocates a buffer on the GPU and initializes it with data.
    pub fn from_slice(context: &Context, data: &[T]) -> Self {
        let buffer = context
            .device
            .handle
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("buffer"),
                contents: bytemuck::cast_slice(data),
                usage: Self::USAGES,
            });

        Self {
            device: Arc::clone(&context.device),
            handle: buffer,
            _marker: PhantomData,
        }
    }

    /// Reads the contents of the buffer into a Vec.
    pub fn read_to_vec(&self) -> Vec<T> {
        let dst_buffer = self.device.handle.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination copy buffer"),
            size: self.handle.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .handle
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy buffer command encoder"),
                });

        encoder.copy_buffer_to_buffer(&self.handle, 0, &dst_buffer, 0, dst_buffer.size());

        self.device.queue.submit(std::iter::once(encoder.finish()));

        let dst_slice = dst_buffer.slice(..);

        dst_slice.map_async(wgpu::MapMode::Read, move |_| {});

        self.device.handle.poll(wgpu::Maintain::Wait);

        let data = dst_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}
