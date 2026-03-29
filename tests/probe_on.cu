// Probe: CSE correctness with pointer arithmetic, struct field reuse,
// and optimizer interaction with inline functions.

// ------------------------------------------------------------------
// CSE: same pointer arithmetic should be CSE'd if operands identical.
// addr = base + tid*4 computed twice in same block — CSE should fold.

__global__ void cse_ptr_arith(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = data[tid];    // base + tid*4
        float b = data[tid];    // same expression — CSE candidate
        out[tid] = a + b;       // should be 2*data[tid] (CSE'd to single load)
    }
}

// ------------------------------------------------------------------
// CSE across struct field reads: reading the same field twice.
// sp->len read twice — second read should be CSE'd if sp not written between.

__device__ float avg_device(int *data, int *len_ptr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += (float)data[i];
    return (n > 0) ? sum / (float)n : 0.0f;
}

__global__ void cse_field_read(float *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = avg_device(data, 0, n);
    }
}

// ------------------------------------------------------------------
// Optimizer + inline: constant folded result from inlined function.
// The inlined body of triple(2) should constant-fold to 6.

__device__ int identity(int x) { return x; }

__global__ void inline_const_fold(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // These should all constant-fold after inlining
        int a = identity(5);       // 5
        int b = identity(a + 3);   // 8
        int c = identity(b * 2);   // 16
        out[0] = c;                // should store constant 16
    }
}

// ------------------------------------------------------------------
// Dead code after return: unreachable instructions after return in device fn.

__device__ int abs_val(int x) {
    if (x < 0) return -x;
    return x;
    // dead code: x += 1;  -- not present, just testing the return path
}

__global__ void abs_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = abs_val(data[tid]);
    }
}

// ------------------------------------------------------------------
// Multiple uses of same computed address in the same block.
// out[tid] written twice with different values — second write wins.

__global__ void double_write(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid];        // first write
        out[tid] = data[tid] + 1;    // second write — overwrites first
    }
}
