// Probe: C++ exceptions, try/catch (should be skipped/ignored in device code),
//        operator overloading syntax (probably should fail gracefully),
//        template class usage (fail gracefully),
//        initializer list constructor (fail gracefully)

// This should work: basic lambda-like device function
__device__ int apply_op(int x, int op) {
    switch (op) {
        case 0: return x + 1;
        case 1: return x - 1;
        case 2: return x * 2;
        case 3: return x / 2;
        default: return x;
    }
}

__global__ void apply_ops(int *out, int *in, int *ops, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = apply_op(in[tid], ops[tid]);
    }
}

// Multiple assignment in declaration (comma-separated same-type decl)
__global__ void multi_decl(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = 1, y = 2, z = 3;
        out[tid] = x + y + z + tid;
    }
}
