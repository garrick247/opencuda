// Liveness test: int, float, and pointer values all live simultaneously.
// Tests that the per-type linear scan buckets don't interfere with each other.
__global__ void mixed_type_pressure(float *out, int *idx, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        int j = idx[tid + 1];
        float v = data[i];
        float w = data[j];
        out[tid] = v + w + (float)(i + j);
    }
}
