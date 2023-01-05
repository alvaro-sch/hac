struct ComputeInput {
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) size: vec3<u32>,
}

@group(0) @binding(0)
var i_sampler: sampler;

@group(1) @binding(0)
var input: texture_2d<f32>;
@group(1) @binding(1)
var output: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0)
var<storage, read> kernel: array<f32>;
@group(2) @binding(1)
var<storage, read> radius: i32;

var<push_constant> direction: vec2<i32>;

@compute @workgroup_size(16, 16)
fn gaussian_pass(in: ComputeInput) {
    let size = vec2<f32>(in.size.xy);
    let id = vec2<i32>(in.id.xy);

    var output_color = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 2 * radius + 1; i++) {
        let current_pixel = vec2<f32>(id + direction * (i - radius));
        let normalized = (current_pixel + 0.5) / size;

        output_color += kernel[i] * textureSampleLevel(input, i_sampler, normalized, 0.0).rgb;
    }

    textureStore(output, id, vec4<f32>(output_color, 1.0));
}
