fn gaussian_kernel_pass(radius: i32, variance: f32) -> Vec<f32> {
    let mut sum_weights = 0.0;

    let mut weights = (-radius..=radius)
        .map(|x| {
            let x_sq = (x as f32).powf(2.0);

            let weight = f32::exp(-x_sq / (2.0 * variance));
            sum_weights += weight;
            weight
        })
        .collect::<Vec<_>>();

    weights.iter_mut().for_each(|x| *x /= sum_weights);

    weights
}

fn main() {
    let input = include_bytes!("polar_bear.jpg");
    let image = image::load_from_memory(input).unwrap().to_rgba8();

    let context = hac::Context::new(&hac::ContextInfo {
        features: hac::Features::PUSH_CONSTANTS,
        limits: hac::Limits {
            max_push_constant_size: 8,
            ..Default::default()
        },
        ..Default::default()
    });

    let sampler = context.sampler(&hac::SamplerInfo {
        mag_filter: hac::FilterMode::Linear,
        min_filter: hac::FilterMode::Linear,
        address_mode_u: hac::AddressMode::ClampToEdge,
        address_mode_v: hac::AddressMode::ClampToEdge,
        ..Default::default()
    });

    // the SamplerBindingType has to be filtering because `FilterMode::Linear` requires it
    let sampler_bind_group = context
        .bind_group_descriptor()
        .push_sampler(&sampler, hac::SamplerBindingType::Filtering)
        .into_bind_group();

    // the image to be sampled with a filtering sampler requires itself to be filterable
    let im0 =
        context.image_from_rgba8_img(&image, hac::ImageSampleType::Float { filterable: true });
    let im1 = hac::Image::empty_like(&im0);

    // 2 bind groups for swapping the 2 images as input and output
    // I'll make a better alternative for this pattern
    let img_bind_group0 = context
        .bind_group_descriptor()
        .push_image(&im0)
        .push_storage_image(&im1, hac::StorageImageAccess::WriteOnly)
        .into_bind_group();

    let img_bind_group1 = context
        .bind_group_descriptor()
        .push_image(&im1)
        .push_storage_image(&im0, hac::StorageImageAccess::WriteOnly)
        .into_bind_group();

    let radius = 10;
    let variance = 5.0;
    let weights = gaussian_kernel_pass(radius, variance);

    let gaussian_kernel_buffer = context.buffer_from_slice(&weights);
    let gaussian_rad_buffer = context.buffer_from_slice(&[radius]);

    let gauss_bind_group = context
        .bind_group_descriptor()
        .push_buffer(&gaussian_kernel_buffer, hac::BufferAccess::ReadOnly)
        .push_buffer(&gaussian_rad_buffer, hac::BufferAccess::ReadOnly)
        .into_bind_group();

    let gaussian_program = context.program_from_wgsl(include_str!("gaussian_blur.wgsl"));

    let gaussian_kernel = context.kernel(&hac::KernelInfo {
        program: &gaussian_program,
        entry_point: "gaussian_pass",
        bind_groups: &[&sampler_bind_group, &img_bind_group0, &gauss_bind_group],
        push_constants_range: Some(0..8),
    });

    let (width, height) = image.dimensions();
    let global_workgroup = hac::Range::d2(width, height);

    context
        .command_queue()
        .enqueue_set_kernel(&gaussian_kernel)
        .enqueue_set_push_constants(0, hac::cast_slice(&[1i32, 0]))
        .enqueue_dispatch(global_workgroup)
        .enqueue_set_bind_group(1, &img_bind_group1)
        .enqueue_set_push_constants(0, hac::cast_slice(&[0i32, 1]))
        .enqueue_dispatch(global_workgroup)
        .execute();

    let output_bytes = im0.read_to_vec();

    image::save_buffer(
        "gaussian_blur_output.png",
        &output_bytes,
        width,
        height,
        image::ColorType::Rgba8,
    )
    .unwrap();
}
