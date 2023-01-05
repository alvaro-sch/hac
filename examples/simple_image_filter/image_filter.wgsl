struct ComputeInput {
    @builtin(global_invocation_id) id: vec3<u32>,
}

@group(0) @binding(0)
var input: texture_2d<f32>;
@group(0) @binding(1)
var output: texture_storage_2d<rgba8unorm, write>;

var<push_constant> color_filter: vec4<f32>;

@compute @workgroup_size(1)
fn main(in: ComputeInput) {
    let sample_coord = vec2<i32>(in.id.xy);

    let pixel = textureLoad(input, sample_coord, 0);
    textureStore(output, sample_coord, color_filter * pixel);
}
