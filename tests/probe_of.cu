// Probe: __device__ global arrays, sizeof in expressions, compound assignment
// on global vars, multiple kernels using same device function.

// ------------------------------------------------------------------
// __device__ global array: indexed read and write from multiple kernels.

__device__ int g_lut[16];

__global__ void lut_init(int base) {
    int tid = threadIdx.x;
    if (tid < 16) {
        g_lut[tid] = base + tid;
    }
}

__global__ void lut_read(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = g_lut[tid & 15];
    }
}

// ------------------------------------------------------------------
// Compound assignment on a global device variable: g_sum += val.
// Tests load-modify-store through GlobalAddrInst.

__device__ int g_sum;

__global__ void accumulate(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            g_sum += data[i];
        }
    }
}

// ------------------------------------------------------------------
// Multiple kernels sharing the same __device__ helper.
// The helper must be inlined correctly into both.

__device__ int triple(int x) { return x * 3; }

__global__ void use_triple_a(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = triple(data[tid]);
}

__global__ void use_triple_b(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = triple(data[tid]) + 1;
}

// ------------------------------------------------------------------
// sizeof in array index computation.
// Tests that sizeof(int) == 4 is folded at compile time.

__global__ void sizeof_index(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof(int) == 4 at compile time — should fold to index*4 in ptx
        int idx = tid * (int)sizeof(int);
        // Write to byte-indexed position (treating out as char*)
        // Actually just use it as a multiplier constant
        out[tid] = data[tid] * (int)sizeof(float);  // * 4
    }
}
