use std::sync::Arc;

pub use wgpu::{Extent3d, ImageDataLayout};

use crate::Context;

pub type ImageFormat = wgpu::TextureFormat;
pub type ImageDimension = wgpu::TextureDimension;
pub type StorageImageAccess = wgpu::StorageTextureAccess;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageSampleType {
    /// textureLoad returns f32s.
    Float { filterable: bool },

    /// textureLoad returns u32s.
    Uint,

    /// textureLoad returns i32s.
    Sint,
}

impl From<ImageSampleType> for wgpu::TextureSampleType {
    fn from(sample_type: ImageSampleType) -> Self {
        match sample_type {
            ImageSampleType::Float { filterable } => wgpu::TextureSampleType::Float { filterable },
            ImageSampleType::Uint => wgpu::TextureSampleType::Uint,
            ImageSampleType::Sint => wgpu::TextureSampleType::Sint,
        }
    }
}

/// Information to create an Image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Size of the image.
    ///
    /// For 2D images set `depht_or_array_layers` to 1.
    pub size: Extent3d,

    /// Format of the image.
    pub format: ImageFormat,
}

/// Handle of an image stored in the GPU.
#[derive(Debug)]
pub struct Image {
    pub(crate) device: Arc<crate::Device>,
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: Extent3d,
    pub(crate) format: ImageFormat,
    pub(crate) dimension: ImageDimension,
}

impl Image {
    // cheesy workaround to be able to make a const bitflag
    // see: https://github.com/bitflags/bitflags/issues/180
    const USAGES: wgpu::TextureUsages = wgpu::TextureUsages::from_bits_truncate(
        wgpu::TextureUsages::TEXTURE_BINDING.bits()
            | wgpu::TextureUsages::STORAGE_BINDING.bits()
            | wgpu::TextureUsages::COPY_DST.bits()
            | wgpu::TextureUsages::COPY_SRC.bits(),
    );

    /// Creates an empty image with the specified info.
    ///
    /// # Note
    ///
    /// The easiest way to submit an image to the GPU is by creating an
    /// [RgbaImage](https://docs.rs/image/latest/image/type.RgbaImage.html)
    /// and using `Image::from_rgba8_image()` (or `Context::image_from_rgba8_img()`)
    /// that is unlocked by enabling the "image" feature.
    pub fn new(context: &Context, info: &ImageInfo) -> Self {
        let dimension = if info.size.depth_or_array_layers == 1 {
            wgpu::TextureDimension::D2
        } else {
            wgpu::TextureDimension::D3
        };

        let texture = context
            .device
            .handle
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Image"),
                usage: Self::USAGES,
                mip_level_count: 1,
                sample_count: 1,
                format: info.format,
                size: info.size,
                dimension,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            view,
            texture,
            dimension,
            size: info.size,
            format: info.format,
            device: Arc::clone(&context.device),
        }
    }

    /// Creates an empty image with the same size and format of the original image.
    pub fn empty_like(original: &Self) -> Self {
        let &Image {
            format,
            size,
            dimension,
            ..
        } = original;

        let texture = original
            .device
            .handle
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Image"),
                usage: Self::USAGES,
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format,
                size,
                view_formats: &[],
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            size,
            view,
            format,
            texture,
            dimension,
            device: Arc::clone(&original.device),
        }
    }

    /// Writes data to an image.
    ///
    /// # Panics
    ///
    /// - if data overruns the size of the image.
    pub fn write(&self, data: &[u8], data_layout: ImageDataLayout, size: Extent3d) {
        self.device.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            data_layout,
            size,
        );
    }

    /// Reads an image to a Vec of bytes.
    pub fn read_to_vec(&self) -> Vec<u8> {
        // KUDOS to @redwarp I struggled to much trying to copy a texture into a buffer
        // https://github.com/redwarp/blog/tree/main/code-sample/image-filters
        let bytes_per_pixel = self.format.describe().block_size as usize;

        let Extent3d { width, height, .. } = self.size;

        let padded_bytes_per_row = {
            let bytes_per_row = bytes_per_pixel * width as usize;
            let padding = (256 - bytes_per_row % 256) % 256;
            bytes_per_row + padding
        };

        let unpadded_bytes_per_row = bytes_per_pixel * width as usize;

        let output_buffer_size =
            padded_bytes_per_row as u64 * height as u64 * std::mem::size_of::<u8>() as u64;

        let dst_buffer = self.device.handle.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Destination copy buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .handle
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy buffer command encoder"),
                });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &dst_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(padded_bytes_per_row as u32),
                    rows_per_image: None,
                },
            },
            self.size,
        );

        self.device.queue.submit(std::iter::once(encoder.finish()));

        let dst_slice = dst_buffer.slice(..);
        dst_slice.map_async(wgpu::MapMode::Read, move |_| {});

        self.device.handle.poll(wgpu::Maintain::Wait);

        let mut pixels = vec![0; unpadded_bytes_per_row * height as usize];

        dst_slice
            .get_mapped_range()
            .chunks_exact(padded_bytes_per_row)
            .zip(pixels.chunks_exact_mut(unpadded_bytes_per_row))
            .for_each(|(padded, pixels)| {
                pixels.copy_from_slice(&padded[..unpadded_bytes_per_row]);
            });

        pixels
    }

    /// Size of the image.
    pub fn size(&self) -> Extent3d {
        self.size
    }

    /// Format of the image.
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    /// Dimension of the image.
    pub fn dimension(&self) -> ImageDimension {
        self.dimension
    }

    #[cfg(feature = "from_image")]
    /// Creates an image from an Rgba8 image buffer.
    ///
    /// The `sample_type` parameter is used to choose the correct image format.
    pub fn from_rgba8_image(
        context: &Context,
        image: &image::RgbaImage,
        sample_type: ImageSampleType,
    ) -> Self {
        let (width, height) = image.dimensions();
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let format = match sample_type {
            ImageSampleType::Float { .. } => ImageFormat::Rgba8Unorm,
            ImageSampleType::Uint => ImageFormat::Rgba8Uint,
            ImageSampleType::Sint => ImageFormat::Rgba8Sint,
        };

        let dimension = ImageDimension::D2;

        let texture = context
            .device
            .handle
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("texture"),
                usage: Self::USAGES,
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format,
                size,
            });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let self_ = Self {
            size,
            view,
            format,
            texture,
            dimension,
            device: Arc::clone(&context.device),
        };

        let bytes_per_pixel = 4;
        self_.write(
            image,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(width * bytes_per_pixel),
                rows_per_image: None,
            },
            size,
        );

        self_
    }
}
