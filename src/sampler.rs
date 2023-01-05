pub use wgpu::{AddressMode, FilterMode, SamplerBindingType, SamplerBorderColor};

use crate::Context;

/// Information to create a sampler.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SamplerInfo {
    /// What to do when sampling out of bounds in the u direction.
    pub address_mode_u: AddressMode,
    /// What to do when sampling out of bounds in the v direction.
    pub address_mode_v: AddressMode,
    /// What to do when sampling out of bounds in the w direction.
    pub address_mode_w: AddressMode,
    /// How to filter when the image has to be magnified.
    pub mag_filter: FilterMode,
    /// How to filter when the image has to be minified.
    pub min_filter: FilterMode,
    /// Color of the border if `AddressMode::ClampToBorder` was chosen.
    pub border_color: Option<SamplerBorderColor>,
}

/// Encodes information to determine the appropiate color that should be
/// returned when sampling an image.
///
/// # Note
///
/// It's forbidden to use `textureSample()` on a copmute stage in wgsl.
/// To be able to sample an image use `textureSampleLevel()` instead.
#[derive(Debug)]
pub struct Sampler {
    pub(crate) handle: wgpu::Sampler,
}

impl Sampler {
    /// Creates a new sampler with the info specified.
    pub fn new(context: &Context, info: &SamplerInfo) -> Self {
        let sampler = context
            .device
            .handle
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Sampler"),
                address_mode_u: info.address_mode_u,
                address_mode_v: info.address_mode_v,
                address_mode_w: info.address_mode_w,
                mag_filter: info.mag_filter,
                min_filter: info.min_filter,
                border_color: info.border_color,
                ..Default::default()
            });

        Self { handle: sampler }
    }
}
