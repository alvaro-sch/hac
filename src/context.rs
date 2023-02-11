use std::sync::Arc;

use bytemuck::Pod;
use pollster::FutureExt as _;
pub use wgpu::{Backends, Dx12Compiler, Features, Limits};

use crate::{
    BindGroupDescriptor, Buffer, CommandQueue, Image, ImageInfo, Kernel, KernelInfo, Program,
    Sampler, SamplerInfo,
};

/// Information to create a context.
#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub backends: Backends,
    pub features: Features,
    pub limits: Limits,
    pub dx12_shader_compiler: Dx12Compiler,
}

impl Default for ContextInfo {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            features: Features::empty(),
            limits: Limits::default(),
            dx12_shader_compiler: Dx12Compiler::default(),
        }
    }
}

/// Manager used to create resources
#[derive(Debug)]
pub struct Context {
    pub(crate) device: Arc<crate::Device>,
}

impl Context {
    /// Creates a context.
    pub fn new(info: ContextInfo) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: info.backends,
            dx12_shader_compiler: info.dx12_shader_compiler,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();

        Self::from_wgpu_adapter(
            &adapter,
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                features: info.features,
                limits: info.limits.clone(),
            },
        )
    }

    /// Creates a context from a wgpu Adapter.
    ///
    /// Useful when wanting to use a specific adapter i.e. one that supports presenting
    /// th the screen, if that's not the case consider `Context::new()`.
    ///
    /// The context will acquire it's own `wgpu::Device` and `wgpu::Queue`.
    pub fn from_wgpu_adapter(
        adapter: &wgpu::Adapter,
        device_descriptor: &wgpu::DeviceDescriptor,
    ) -> Self {
        let (device, queue) = adapter
            .request_device(&device_descriptor, None)
            .block_on()
            .unwrap();

        Self {
            device: Arc::new(crate::Device {
                handle: device,
                queue,
            }),
        }
    }

    /// Creates an empty buffer capable of holding `capacity` **elements of T**.
    ///
    /// # Panics
    ///
    /// - if `capacity * std::mem::size_of::<T>()` exceeds the `max_buffer_size` limit
    /// set in [`ContextInfo`] (with a default of 2^30).
    pub fn buffer<T: Pod>(&self, capacity: wgpu::BufferAddress) -> Buffer<T> {
        Buffer::new(self, capacity)
    }

    /// Creates an buffer initialized from a slice.
    ///
    /// # Panics
    ///
    /// - if `std::mem::size_of_val(data)` exceeds the `max_buffer_size` limit
    /// set in [`ContextInfo`] (with a default of 2^30).
    pub fn buffer_from_slice<T: Pod>(&self, data: &[T]) -> Buffer<T> {
        Buffer::from_slice(self, data)
    }

    /// Creates an [`Image`] with info.
    pub fn image(&self, info: &ImageInfo) -> Image {
        Image::new(self, info)
    }

    /// Creates a [`Sampler`] with info.
    pub fn sampler(&self, info: &SamplerInfo) -> Sampler {
        Sampler::new(self, info)
    }

    /// Creates a [`BindGroupDescriptor`] (a.k.a. descriptor set) to bind resources
    /// such as buffers, samplers and images.
    pub fn bind_group_descriptor(&self) -> BindGroupDescriptor {
        BindGroupDescriptor::new(self)
    }

    /// Creates a [`Program`] from a `wgpu::ShaderSource`.
    pub fn program_from_shader_source(&self, source: wgpu::ShaderSource) -> Program {
        Program::from_source(self, source)
    }

    /// Creates a [`Program`] from wgsl source code.
    pub fn program_from_wgsl(&self, source: &str) -> Program {
        let shader_source = wgpu::ShaderSource::Wgsl(source.into());
        self.program_from_shader_source(shader_source)
    }

    /// Creates a [`Kernel`] with info.
    pub fn kernel(&self, info: &KernelInfo) -> Kernel {
        Kernel::new(self, info)
    }

    /// Creates a [`CommandQueue`].
    pub fn command_queue(&self) -> CommandQueue {
        CommandQueue::new(self)
    }

    #[cfg(feature = "from_image")]
    /// Creates an image from an RgbaImage of the image crate.
    pub fn image_from_rgba8_img(
        &self,
        image: &image::RgbaImage,
        sample_type: crate::ImageSampleType,
    ) -> Image {
        Image::from_rgba8_image(self, image, sample_type)
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new(ContextInfo::default())
    }
}
