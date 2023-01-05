use std::{collections::VecDeque, sync::Arc};

use crate::{BindGroup, Context, Kernel, Range};

/// Avaiable commands to execute in a CommandQueue.
#[derive(Debug)]
pub enum Command<'a> {
    /// Set a kernel to be able to set PushConstants or Dispatch it.
    SetKernel { kernel: &'a Kernel },

    /// Set push constants in the range `offset`..`data.len()`
    ///
    /// Requires a kernel to be set beforehand and the `PUSH_CONSTANT` feature.
    SetPushConstants { offset: u32, data: &'a [u8] },

    /// Sets a bind group at `index`.
    ///
    /// It's sometimes better to pre-create a set of bind groups with the same
    /// layout rather than writing the buffers they point to.
    SetBindGroup {
        index: u32,
        bind_group: &'a BindGroup,
    },

    /// Dispatch a previously set Kernel with `workgroups` workgroup sizes.
    ///
    /// Requires a kernel to be set beforehand.
    Dispatch { workgroups: Range },
}

/// Queue that holds Commands and executes them in FIFO order.
#[derive(Debug)]
pub struct CommandQueue<'a> {
    pub(crate) device: Arc<crate::Device>,
    pub(crate) cmd_queue: VecDeque<Command<'a>>,
}

impl<'a> CommandQueue<'a> {
    /// Creates an empty command queue from a [`Context`].
    pub fn new(context: &Context) -> Self {
        Self {
            device: Arc::clone(&context.device),
            cmd_queue: VecDeque::new(),
        }
    }

    /// Enqueue a [`Kernel`].
    ///
    /// Does not execute the kernel right away to be able to bind other
    /// resources (i.e. push constants).
    ///
    /// To execute an already set kernel see `CommandQueue::enqueue_dispatch()`.
    pub fn enqueue_set_kernel(mut self, kernel: &'a Kernel) -> Self {
        self.cmd_queue.push_back(Command::SetKernel { kernel });
        self
    }

    /// Enqueue push constants in the range `offset`..`data.len()`.
    /// - both `offset` and `data.len()` must always be a multiple of 4.
    ///
    /// # Note
    ///
    /// To be able to use push constants the `PUSH_CONSTANTS` feature must be enabled
    /// along with setting the correct limits in [`ContextInfo`]. The program will panic
    /// otherwise when executing the queue.
    pub fn enqueue_set_push_constants(mut self, offset: u32, data: &'a [u8]) -> Self {
        self.cmd_queue
            .push_back(Command::SetPushConstants { offset, data });
        self
    }

    /// Enqueue setting a bind group at a certain index.
    ///
    /// This may be used for example when the same kernel has to run multiple times
    /// with different input (and that input is large enough so that push constants
    /// can't be used).
    ///
    /// # Note
    ///
    /// The bind group to be set must have the same layout as the one that was set
    /// when the currently bound kernel was created, the program will panic when executing
    /// the queue otherwise.
    pub fn enqueue_set_bind_group(mut self, index: u32, bind_group: &'a BindGroup) -> Self {
        self.cmd_queue
            .push_back(Command::SetBindGroup { index, bind_group });
        self
    }

    /// Enqueues a dispatch command on a set kernel.
    ///
    /// # Note
    ///
    /// Each dimension must not exceed the limit size `max_compute_workgroups_per_dimension`
    /// with a default value of 65535 that can be configured in `ContextInfo`.
    pub fn enqueue_dispatch(mut self, workgroups: Range) -> Self {
        self.cmd_queue.push_back(Command::Dispatch { workgroups });
        self
    }

    /// Executes the Commands recorded in the queue.
    ///
    /// # Panics
    ///
    /// - if `Command::Dispatch` was enqueued before setting a kernel.
    /// - if `Command::SetPushConstants` was enqueued before setting a kernel.
    /// - if `Command::SetPushConstants` is used without enabling the `PUSH_CONSTANTS` feature
    /// or exceeds the maximum set limit specified in [`ContextInfo`].
    /// - if `Command::SetPushConstants` is used twice for the same Kernel.
    /// - if `Command::SetBindGroup` is bound at an index which is supposed to have a bind group
    /// with a different layout.
    pub fn execute(self) {
        let mut encoder =
            self.device
                .handle
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command encoder"),
                });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute pass"),
        });

        self.cmd_queue
            .into_iter()
            .for_each(|command| compute_pass.execute(command));

        drop(compute_pass);

        self.device.queue.submit(std::iter::once(encoder.finish()));
    }
}

trait ExecuteCommand<'a> {
    fn execute(&mut self, command: Command<'a>);
}

impl<'a, 'b> ExecuteCommand<'b> for wgpu::ComputePass<'a>
where
    'b: 'a,
{
    fn execute(&mut self, command: Command<'b>) {
        match command {
            Command::SetPushConstants { offset, data } => self.set_push_constants(offset, data),

            Command::SetKernel { kernel } => {
                self.set_pipeline(&kernel.pipeline);

                kernel
                    .bind_groups
                    .iter()
                    .enumerate()
                    .for_each(|(i, bind_group)| {
                        self.set_bind_group(i as u32, bind_group, &[]);
                    });
            }

            Command::SetBindGroup { index, bind_group } => {
                self.set_bind_group(index, &bind_group.handle, &[]);
            }

            Command::Dispatch { workgroups } => {
                let Range { x, y, z } = workgroups;
                self.dispatch_workgroups(x, y, z);
            }
        }
    }
}
