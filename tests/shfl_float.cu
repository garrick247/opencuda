// Regression: __shfl_*_sync with float arg must infer return type as float,
// NOT INT32. Without the fix, the shuffled float bits get corrupted by
// cvt.rn.f32.s32 (numeric conversion) instead of being preserved bitwise.
__global__ void shfl_float_reduce(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Warp-level float reduction via shuffle
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    if (tid == 0) {
        out[0] = val;
    }
}
