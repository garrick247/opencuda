// Probe: Patterns at the boundary of what the IR can represent
// - Empty kernel body
// - Kernel with only a return statement
// - Kernel where all branches return early (no fall-through path)
// - Kernel that only does shared memory operations
// - Kernel with a single unconditional write

__global__ void empty_kernel(int *out, int n) {
    // Intentionally empty - should produce minimal PTX
}

__global__ void single_store(int *out, int val) {
    if (threadIdx.x == 0) {
        out[0] = val;
    }
}

// All paths return early
__global__ void all_early_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    if (in[tid] < 0) {
        out[tid] = -1;
        return;
    }
    if (in[tid] == 0) {
        out[tid] = 0;
        return;
    }
    out[tid] = 1;
}
