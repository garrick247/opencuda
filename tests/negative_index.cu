// Test: pointer arithmetic with a negative offset (stencil-style).
// Must correctly compute base + (i-1)*sizeof(float) even when tid=0.
__global__ void negative_index(float *out, float *in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > 0 && tid < n) {
        // Access in[tid-1]: this tests negative-ish address computation
        out[tid] = in[tid] - in[tid - 1];
    }
}
