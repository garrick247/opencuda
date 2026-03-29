// Probe: Edge cases in the LICM optimizer
// - Loop with invariant that depends on an array load (not safe to hoist without alias analysis)
// - Loop with invariant pure computation that IS safe to hoist
// - Loop with multiple invariants, some dependent on others
// - Loop where hoisting would be wrong due to __shared__ dependency

__global__ void licm_safe(float *out, float *in, float scale, float bias, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // scale and bias are loop-invariant params (safe to fold once)
        float inv_scale = 1.0f / scale;  // invariant
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += (in[(tid + i) % n] + bias) * inv_scale;
        }
        out[tid] = sum;
    }
}

__global__ void licm_array_dep(float *out, float *in, float *weights, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        // weights[0] is loop-invariant (same address each iter)
        // but LICM should only hoist if no aliasing — our LICM is conservative
        for (int i = 0; i < 8; i++) {
            sum += in[(tid + i) % n] * weights[0];
        }
        out[tid] = sum;
    }
}
