// Probe: Patterns that stress the SSA IR representation
// - Value used in condition that's also modified in the true branch
// - Same variable appearing as both input and output
// - Chain of dependent assignments

__global__ void ssa_stress(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        // x used in condition AND modified in both branches
        if (x > 0) {
            x = x - 1;  // x is modified here
        } else {
            x = x + 1;  // and here
        }
        // After merge, x has the value from whichever branch ran
        out[tid] = x;
    }
}

// Long dependency chain
__global__ void dep_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        v = v * 3 + 1;
        v = v / 2 + v % 2;
        v = (v > 100) ? v - 100 : v;
        v = v ^ (v >> 4);
        v = v & 0xFF;
        out[tid] = v;
    }
}

// In-place update pattern (a = a op b)
__global__ void inplace_update(int *arr, int *delta, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        arr[tid] = arr[tid] + delta[tid];  // read and write same location
    }
}
