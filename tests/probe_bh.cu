// Probe: PTX correctness issues — validate with ptxas
// These patterns should produce correct PTX, not just non-crashing PTX

// 1. Store to __shared__ then barrier then load
__global__ void shared_barrier_roundtrip(float *out, float *in, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    smem[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    // Each thread reads neighbor's value
    int neighbor = (tid + 1) % blockDim.x;
    float v = smem[neighbor];
    if (tid < n) out[tid] = v;
}

// 2. Condition variable that's a float comparison
__global__ void float_cond(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = 0.0f;
        if (v != 0.0f) {
            r = 1.0f / v;
        }
        out[tid] = r;
    }
}

// 3. Long chain of arithmetic (register pressure)
__global__ void deep_chain(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = v * 1.1f + 0.5f;
        float b = a * a - v;
        float c = sqrtf(b * b + a * a);
        float d = c / (a + 1.0f);
        float e = d * v + c - b;
        float f2 = e * e - d * d;
        float g = sqrtf(fabsf(f2));
        out[tid] = g + a + b + c + d + e;
    }
}
