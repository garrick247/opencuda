// Nasty: half-precision accumulator across a for-loop back-edge.
// Tests that the loop writeback correctly handles .f16 Values.
__global__ void half_sum(half* data, float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    half acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc = acc + data[i];
    }
    // Widen to f32 for output
    out[tid] = acc;
}

// Second: half inside condition — tests half comparison lowering.
__global__ void half_threshold(half* data, half* out, int n, half thresh) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    half v = data[tid];
    if (v > thresh) {
        out[tid] = v;
    } else {
        out[tid] = thresh;
    }
}
