use std::sync::Arc;

use crate::{BindGroup, CommandQueue, Context, Range};

/// Wrapper of a `wgpu::ShaderModule`.
#[derive(Debug)]
pub struct Program(wgpu::ShaderModule);

impl Program {
    /// Creates a Program from a `wgpu::ShaderSource`.
    ///
    /// [`Context`] provides more ergonomic methods for creating a program
    /// (i.e `Context::program_from_wgsl()`).
    pub fn from_source(context: &Context, source: wgpu::ShaderSource) -> Self {
        let shader = context
            .device
            .handle
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source,
            });

        Self(shader)
    }
}

/// Provides the info required to execute a kernel.
#[derive(Debug)]
pub struct KernelInfo<'a> {
    /// Program that will be executed.
    pub program: &'a Program,

    /// Function of the program that will run when the kernel is dispatched.
    pub entry_point: &'a str,

    /// Handles that hold the buffers, textures and samplers to be used in the kernel.
    pub bind_groups: &'a [&'a BindGroup],

    /// Range of a small data that can be cheaply changed on every kernel dispatch.
    pub push_constants_range: Option<std::ops::Range<u32>>,
}

/// Program that executes on the device.
#[derive(Debug)]
pub struct Kernel {
    pub(crate) device: Arc<crate::Device>,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_groups: Vec<Arc<wgpu::BindGroup>>,
}

impl Kernel {
    /// Creates a kernel.
    pub fn new(context: &Context, info: &KernelInfo) -> Self {
        let device = Arc::clone(&context.device);

        let num_entries = info.bind_groups.len();

        let mut layouts = Vec::with_capacity(num_entries);
        let mut bind_groups = Vec::with_capacity(num_entries);

        info.bind_groups.iter().for_each(|bind_group| {
            layouts.push(&bind_group.layout);
            bind_groups.push(Arc::clone(&bind_group.handle));
        });

        let is_some = info.push_constants_range.is_some() as usize;
        let push_constant_ranges = &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: info.push_constants_range.clone().unwrap_or(0..0),
        }][0..is_some];

        let pipeline_layout =
            device
                .handle
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline layout"),
                    bind_group_layouts: &layouts,
                    push_constant_ranges,
                });

        let pipeline = device
            .handle
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &info.program.0,
                entry_point: info.entry_point,
            });

        Self {
            device,
            pipeline,
            bind_groups,
        }
    }

    /// Executes a kernel.
    ///
    /// It's a nice shortcut when only needing to run it once without caring about
    /// binding things like push constants.
    ///
    /// If that's not the intention then check [`CommandQueue`].
    pub fn dispatch(&self, workgroups: Range) {
        let command_queue = CommandQueue {
            device: Arc::clone(&self.device),
            cmd_queue: std::collections::VecDeque::new(),
        };

        command_queue
            .enqueue_set_kernel(self)
            .enqueue_dispatch(workgroups)
            .execute();
    }
}
