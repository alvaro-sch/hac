use std::sync::Arc;

use crate::{
    Buffer, BufferAccess, Context, Image, ImageDimension, Sampler, SamplerBindingType,
    StorageImageAccess,
};

/// Represents a [`Buffer`]
#[derive(Debug)]
struct BufferBinding<'a> {
    resource: wgpu::BindingResource<'a>,
    access: BufferAccess,
}

impl<'a> From<&BufferBinding<'a>> for wgpu::BindingType {
    fn from(binding: &BufferBinding<'a>) -> Self {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage {
                read_only: binding.access == BufferAccess::ReadOnly,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        }
    }
}

/// Represents an [`Image`] for sampling.
#[derive(Debug)]
struct TextureBinding<'a> {
    resource: wgpu::BindingResource<'a>,
    dimension: wgpu::TextureViewDimension,
    sample_type: wgpu::TextureSampleType,
}

impl<'a> From<&TextureBinding<'a>> for wgpu::BindingType {
    fn from(binding: &TextureBinding<'a>) -> Self {
        wgpu::BindingType::Texture {
            sample_type: binding.sample_type,
            view_dimension: binding.dimension,
            multisampled: false,
        }
    }
}

/// Represents an [`Image`] for storing.
#[derive(Debug)]
struct StorageTextureBinding<'a> {
    resource: wgpu::BindingResource<'a>,
    access: wgpu::StorageTextureAccess,
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureViewDimension,
}

impl<'a> From<&StorageTextureBinding<'a>> for wgpu::BindingType {
    fn from(binding: &StorageTextureBinding<'a>) -> Self {
        wgpu::BindingType::StorageTexture {
            access: binding.access,
            format: binding.format,
            view_dimension: binding.dimension,
        }
    }
}

/// Represents a [`Sampler`].
#[derive(Debug)]
struct SamplerBinding<'a> {
    resource: wgpu::BindingResource<'a>,
    binding_type: wgpu::SamplerBindingType,
}

impl<'a> From<&SamplerBinding<'a>> for wgpu::BindingType {
    fn from(binding: &SamplerBinding<'a>) -> Self {
        wgpu::BindingType::Sampler(binding.binding_type)
    }
}

/// Everything that can be bound to a `wgpu::BindGroup`.
#[derive(Debug)]
enum Binding<'a> {
    Buffer(BufferBinding<'a>),
    Sampler(SamplerBinding<'a>),
    Texture(TextureBinding<'a>),
    StorageTexture(StorageTextureBinding<'a>),
}

impl<'a> Binding<'a> {
    fn into_resource(self) -> wgpu::BindingResource<'a> {
        match self {
            Binding::Buffer(buffer_binding) => buffer_binding.resource,
            Binding::Sampler(sampler_binding) => sampler_binding.resource,
            Binding::Texture(texture_binding) => texture_binding.resource,
            Binding::StorageTexture(storage_texture_binding) => storage_texture_binding.resource,
        }
    }
}

impl<'a> From<&Binding<'a>> for wgpu::BindingType {
    fn from(binding: &Binding<'a>) -> Self {
        match binding {
            Binding::Buffer(buffer_binding) => buffer_binding.into(),
            Binding::Sampler(sampler_binding) => sampler_binding.into(),
            Binding::Texture(texture_binding) => texture_binding.into(),
            Binding::StorageTexture(storage_texture_binding) => storage_texture_binding.into(),
        }
    }
}

/// Contains the information to create BindGroups.
///
/// This may change in the future to be able to reutilize `wgpu::BindGroupLayout`s.
#[derive(Debug)]
pub struct BindGroupDescriptor<'a> {
    device: Arc<crate::Device>,
    bindings: Vec<Binding<'a>>,
}

impl<'a> BindGroupDescriptor<'a> {
    /// Creates an empty bind group layout.
    pub fn new(context: &Context) -> Self {
        Self {
            device: Arc::clone(&context.device),
            bindings: Vec::new(),
        }
    }

    /// Pushes `buffer` as the last binding with `accessor` access.
    ///
    /// # Example wgsl syntax
    /// ```cpp,ignore
    /// @group(X) @binding(Y)
    /// var<storage, 'access'> buffer: array<'T'>; // T is the type of the buffer
    /// ```
    pub fn push_buffer<T>(mut self, buffer: &'a Buffer<T>, access: BufferAccess) -> Self {
        let binding = Binding::Buffer(BufferBinding {
            resource: buffer.handle.as_entire_binding(),
            access,
        });

        self.bindings.push(binding);
        self
    }

    /// Pushes `sampler` as the last binding with the spacified `binding_type`.
    ///
    /// The `binding_type` should be filtering if it uses `FilterMode::Linear`.
    ///
    /// # Example wgsl syntax
    /// ```cpp,ignore
    /// @group(X) @binding(Y)
    /// var i_sampler: sampler;
    ///
    /// // valid usage
    /// let pixel = textureSampleLevel(texture, i_sampler, uv, 0.0);
    ///
    /// // invalid: wgsl doesn't allow textureSample in compute stages
    /// let pixel = textureSample(texture, i_sample, level);
    /// ```
    pub fn push_sampler(mut self, sampler: &'a Sampler, binding_type: SamplerBindingType) -> Self {
        let binding = Binding::Sampler(SamplerBinding {
            resource: wgpu::BindingResource::Sampler(&sampler.handle),
            binding_type,
        });

        self.bindings.push(binding);
        self
    }

    /// Pushes `image` as the last binding.
    ///
    /// # Example wgsl syntax
    /// ```cpp,ignore
    /// @group(X) @binding(Y)
    /// var image: texture_'Nd'<'T'>;
    /// // T is the format of the image:
    /// // - if it's format ends with Unorm => T is f32
    /// // - if it ends with Uint => T is u32
    /// // - if it ends with Sint => T is i32
    /// ```
    pub fn push_image(mut self, image: &'a Image) -> Self {
        let dimension = if image.dimension == ImageDimension::D2 {
            wgpu::TextureViewDimension::D2
        } else {
            wgpu::TextureViewDimension::D3
        };

        let sample_type = image.format.describe().sample_type;

        let binding = Binding::Texture(TextureBinding {
            dimension,
            sample_type,
            resource: wgpu::BindingResource::TextureView(&image.view),
        });

        self.bindings.push(binding);
        self
    }

    /// Pushes an image for storage.
    ///
    /// # Example wgsl syntax
    /// ```cpp,ignore
    /// @group(X) @binding(Y)
    /// var image: texture_storage_2d<rgba8unorm, write>;
    /// ```
    pub fn push_storage_image(mut self, image: &'a Image, access: StorageImageAccess) -> Self {
        let dimension = if image.dimension == ImageDimension::D2 {
            wgpu::TextureViewDimension::D2
        } else {
            wgpu::TextureViewDimension::D3
        };

        let binding = Binding::StorageTexture(StorageTextureBinding {
            access,
            dimension,
            resource: wgpu::BindingResource::TextureView(&image.view),
            format: image.format,
        });

        self.bindings.push(binding);
        self
    }

    /// Creates a bind group.
    pub fn into_bind_group(self) -> BindGroup {
        let num_entries = self.bindings.len();

        let mut layout_entries = Vec::with_capacity(num_entries);
        let mut bind_group_entries = Vec::with_capacity(num_entries);

        self.bindings
            .into_iter()
            .enumerate()
            .for_each(|(i, binding)| {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::from(&binding),
                    count: None,
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: binding.into_resource(),
                })
            });

        let layout =
            self.device
                .handle
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bind group layout"),
                    entries: &layout_entries,
                });

        let bind_group = Arc::new(self.device.handle.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &layout,
                entries: &bind_group_entries,
            },
        ));

        BindGroup {
            layout,
            handle: bind_group,
        }
    }
}

/// Hold the data necesary to set bind groups (a.k.a. descriptor sets) in the Kernel.
///
/// bind groups are created from [`BindGroupDescriptor`]s.
#[derive(Debug)]
pub struct BindGroup {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) handle: Arc<wgpu::BindGroup>,
}
