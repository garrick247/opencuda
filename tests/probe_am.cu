// Probe: PTX correctness — check that output PTX is semantically valid
// - Store to __constant__ variable (should fail gracefully — can't store to const)
// - Store to read-only param (should gracefully emit store anyway)
// - Predicate value used in arithmetic
// - Very large constant values
// - Zero-size array (edge case)
// - Using blockDim.y, blockDim.z, gridDim.y, gridDim.z

__global__ void grid_dims(int *out, int n) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid_z = threadIdx.z + blockIdx.z * blockDim.z;
    int flat = tid_x + tid_y * gridDim.x * blockDim.x
                     + tid_z * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    if (flat < n) {
        out[flat] = flat;
    }
}

__global__ void large_const(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = 0x7FFFFFFFFFFFFFFFLL;
        long long u = -9223372036854775807LL - 1LL;  // INT64_MIN
        out[tid] = v + (long long)tid + u;
    }
}

__global__ void pred_arith(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid];
        // Boolean result in arithmetic context
        int gt = (va > vb) ? 1 : 0;
        int lt = (va < vb) ? 1 : 0;
        int eq = (va == vb) ? 1 : 0;
        out[tid] = gt * 4 + lt * 2 + eq;
    }
}
