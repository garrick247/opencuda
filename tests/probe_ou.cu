// Probe: loop unrolling edge cases — unroll with struct field carry,
// unroll with function call inside, unroll with break, and
// unroll trip count at boundary (exactly 16, 17 iterations).

// ------------------------------------------------------------------
// Loop with exactly UNROLL_MAX (16) iterations — should fully unroll.

__global__ void unroll_16(float *out, float *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Loop with 17 iterations — should NOT unroll (> 16 limit).
// The loop body executes 17 times at runtime.

__global__ void no_unroll_17(float *out, float *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 17; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Unrolled loop with struct field carry.
// The struct field `acc` is loop-carried and must be handled correctly.

struct Acc { float val; int count; };

__global__ void unroll_struct_carry(float *out, float *data, float threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc a;
        a.val = 0.0f;
        a.count = 0;
        for (int i = 0; i < 8; i++) {
            if (data[i] > threshold) {
                a.val += data[i];
                a.count++;
            }
        }
        out[0] = a.val;
        out[1] = (float)a.count;
    }
}

// ------------------------------------------------------------------
// Loop with inline device function call — unroller must remap call args.

__device__ float square(float x) { return x * x; }

__global__ void unroll_with_call(float *out, float *data) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            sum += square(data[i]);
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Loop that computes stride-2 sum: only even indices.
// Tests that i+=2 step is handled correctly.

__global__ void stride2_sum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i += 2) {
            sum += data[i];
        }
        out[0] = sum;
    }
}
