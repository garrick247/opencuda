// Probe: optimizer stress — patterns that might trigger incorrect
// CSE (cross-block fusion), wrong dead code elimination, identity
// fold on float (x-x=0, x/x=1), and LICM interaction with stores.

// ------------------------------------------------------------------
// CSE should NOT cross block boundaries: same expression in two
// different blocks must NOT be merged (would be wrong if values differ).

__global__ void no_cross_block_cse(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r1, r2;
        if (v > 0.0f) {
            r1 = v * v + 1.0f;  // v*v in if-true
        } else {
            r2 = v * v - 1.0f;  // v*v in if-false (different block, v could differ)
            r1 = r2 * 2.0f;
        }
        out[tid] = r1;
    }
}

// ------------------------------------------------------------------
// Identity fold: x - x = 0, x / x = 1.0f for float (not safe for NaN/Inf).
// Verify these do NOT get identity-folded for float (only safe for int).

__global__ void no_float_identity_fold(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // These MUST compute actual values, not fold to 0.0f or 1.0f
        // because v could be NaN or Inf.
        float sub_self = v - v;  // should NOT fold to 0.0f for float
        float mul_one  = v * 1.0f;
        out[tid] = sub_self + mul_one;
    }
}

// ------------------------------------------------------------------
// LICM: a load from a loop-invariant address should stay inside loop
// if address computation depends on loop variable.

__global__ void licm_dependent_addr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            // Address depends on i — must NOT be hoisted
            float v = in[i * 2];
            sum += v;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Dead code that really is dead: store to a local var never read.
// Verify the store is NOT incorrectly elided when the var IS read.

__global__ void no_dead_store_elision(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid];
        float y = x * 2.0f;     // used
        float z = y + 1.0f;     // used
        float w = z * z;        // used
        // w is actually written to output; do NOT elide chain
        out[tid] = w;
    }
}

// ------------------------------------------------------------------
// Constant propagation through phi: after loop exits with known value.

__global__ void const_after_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        while (i < n) i++;
        // i == n here (loop-exit constant if n is known, but n is a param)
        out[0] = i;
    }
}

// ------------------------------------------------------------------
// CSE on repeated pointer arithmetic: same address computed multiple times.

__global__ void repeated_addr_cse(float *out, float *in, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = indices[tid];
        // Load same element 3 times with same index — CSE should deduplicate
        float a = in[idx];
        float b = in[idx] * 2.0f;
        float c = in[idx] + a;
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Optimizer must not remove a store that feeds a later load.

__global__ void store_feeds_load(float *out, float *in, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    // Store to shared
    if (tid < n) smem[lane] = in[tid] * 2.0f;
    __syncthreads();
    // Load from shared (the store above feeds this)
    if (tid < n) {
        int nbr = (lane + 1) % 32;
        out[tid] = smem[nbr];
    }
}
