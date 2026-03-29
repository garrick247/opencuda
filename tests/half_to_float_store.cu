// Test: implicit half → float* store and float → half* store (implicit widening/narrowing).
__global__ void half_to_float_store(float *out, half *a, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    half v = a[tid];
    // Implicit half → float store (must emit cvt.f32.f16)
    out[tid] = v;
}

__global__ void float_to_half_store(half *out, float *a, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    float v = a[tid];
    // Implicit float → half store (must emit cvt.rn.f16.f32)
    out[tid] = v;
}
