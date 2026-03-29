// Probe: gridDim, unsigned literals, math intrinsics, atomicSub
// Tests correctness of CUDA builtins and type system edges.

// ------------------------------------------------------------------
// gridDim access: all six grid/block dimensions must be available.
// Tests that gridDim.x/y/z map to correct PTX registers.

__global__ void grid_info(int *out) {
    out[0] = gridDim.x;
    out[1] = gridDim.y;
    out[2] = gridDim.z;
    out[3] = blockDim.x;
    out[4] = blockDim.y;
    out[5] = blockDim.z;
    out[6] = blockIdx.x;
    out[7] = blockIdx.y;
    out[8] = threadIdx.x;
    out[9] = threadIdx.y;
    out[10] = threadIdx.z;
}

// ------------------------------------------------------------------
// Unsigned integer literals: 0xFFFFFFFFu should be UINT32 (not INT32),
// and large hex (> INT32_MAX) should also be unsigned.

__global__ void unsigned_literals(unsigned int *out, unsigned int *a) {
    int tid = threadIdx.x;
    unsigned int mask = 0xFFFFFFFFu;
    unsigned int big  = 0x80000000u;
    out[0] = a[tid] & mask;
    out[1] = a[tid] & big;
    out[2] = a[tid] + 0xFFFFFFFFu;  // wraps as unsigned
}

// ------------------------------------------------------------------
// Math intrinsics: sqrtf, fminf, fmaxf, fabsf must emit valid PTX
// without silent drops.

__global__ void math_ops(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid*4+0] = sqrtf(a[tid]);
        out[tid*4+1] = fminf(a[tid], b[tid]);
        out[tid*4+2] = fmaxf(a[tid], b[tid]);
        out[tid*4+3] = fabsf(a[tid] - b[tid]);
    }
}

// ------------------------------------------------------------------
// atomicSub: must emit atom.global.add with negated value (PTX has no atom.sub)

__global__ void atomic_sub_test(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicSub(out, data[tid]);
    }
}
