use rand::Rng;

// wgpu's default `max_workgroups_per_dimension`
// can be changed using `hac::Limits` on Context creation
const N: usize = 1 << 16 - 1;

const KERNEL_SOURCE: &'static str = r#"
struct ComputeInput {
    // wgsl builtin variables can be found in the following link
    // https://www.w3.org/TR/WGSL/#builtin-values
    @builtin(global_invocation_id) id: vec3<u32>,
}

@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;

@compute @workgroup_size(1)
fn main(input: ComputeInput) {
    let i = input.id.x;
    c[i] = a[i] + b[i];
}"#;

fn main() {
    let context = hac::Context::new(&hac::ContextInfo::default());

    let mut rng = rand::thread_rng();

    let mut a = vec![0.0f32; N];
    rng.fill(&mut a[..]);

    let mut b = vec![0.0f32; N];
    rng.fill(&mut b[..]);

    let buf_a = context.buffer_from_slice(&a); // input
    let buf_b = context.buffer_from_slice(&b); // input
    let buf_c = context.buffer::<f32>(N as u64); // output

    let bind_group = context
        .bind_group_descriptor()
        .push_buffer(&buf_a, hac::BufferAccess::ReadOnly) // @binding(0)
        .push_buffer(&buf_b, hac::BufferAccess::ReadOnly) // @binding(1)
        .push_buffer(&buf_c, hac::BufferAccess::ReadWrite) // @binding(2)
        .into_bind_group();

    let program = context.program_from_wgsl(KERNEL_SOURCE);

    let kernel = context.kernel(&hac::KernelInfo {
        program: &program,
        entry_point: "main",
        bind_groups: &[&bind_group], // each index corresponds to the group
        // each binding of `bind_group` is in @group(0)
        push_constants_range: None, // requires the `PUSH_CONSTANTS` feature
    });

    kernel.dispatch(hac::Range::d1(N as u32));

    let c = buf_c.read_to_vec(); // read result

    // check if the sums were performed correctly and print some results
    (0..N).for_each(|i| assert!((a[i] + b[i] - c[i]).abs() <= f32::EPSILON));
    (0..8).for_each(|i| println!("{:<11} + {:<11} = {}", a[i], b[i], c[i]));
}
