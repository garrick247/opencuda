// Nasty: early return inside a for-loop (tests return-inside-loop CFG shape).
// The loop writeback in inc_bb must not corrupt the return path.
__global__ void first_positive(float* data, float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    for (int i = 0; i < n; i++) {
        float v = data[i];
        if (v > 0.0f) {
            out[tid] = v;
            return;              // early exit from inside loop
        }
    }
    out[tid] = 0.0f;             // no positive found
}
