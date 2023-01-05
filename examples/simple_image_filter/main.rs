fn main() {
    let input = include_bytes!("polar_bear.jpg");
    let image = image::load_from_memory(input).unwrap().to_rgba8();

    let context = hac::Context::new(&hac::ContextInfo {
        backends: hac::Backends::all(),
        // required to be able to use push constants
        features: hac::Features::PUSH_CONSTANTS,
        // pushing 4 f32s = 16 bytes
        limits: hac::Limits {
            max_push_constant_size: 16,
            ..Default::default()
        },
    });

    // ImageSampleType determines if the texture type will be <f32> <i32> or <u32> in the kernel
    // the filterable is not needed unless a texture sampler is used
    let in_image =
        context.image_from_rgba8_img(&image, hac::ImageSampleType::Float { filterable: false });
    let out_image = hac::Image::empty_like(&in_image);

    let bind_group = context
        .bind_group_descriptor()
        .push_image(&in_image) // @binding(0)
        // output images have the texture_storage_* type instead of texture_* for input images
        .push_storage_image(&out_image, hac::StorageImageAccess::WriteOnly) // @binding(1)
        .into_bind_group();

    let program = context.program_from_wgsl(include_str!("image_filter.wgsl"));
    let kernel = context.kernel(&hac::KernelInfo {
        program: &program,
        entry_point: "main",
        bind_groups: &[&bind_group],       // @group(0)
        push_constants_range: Some(0..16), // offset = 0, size = 16
    });

    let (width, height) = image.dimensions();

    context
        .command_queue()
        // must set the kernel before setting push constants
        .enqueue_set_kernel(&kernel)
        // push constants may have any shape that satisfy their specified range and are pushed as bytes
        // the `cast_slice` function from bytemuck is re-exported for convenience
        .enqueue_set_push_constants(0, hac::cast_slice(&[1.0f32, 1.0, 0.0, 1.0]))
        .enqueue_dispatch(hac::Range::d2(width, height))
        .execute();

    let output_bytes = out_image.read_to_vec();

    image::save_buffer(
        "simple_image_filter_output.png",
        &output_bytes,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
