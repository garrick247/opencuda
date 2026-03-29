// Probe: comparison type promotion + shfl float + signed pointer index widening
// Tests that unsigned comparison uses setp.lt.u32 not setp.lt.s32,
// that __shfl_sync with float uses bit-preserving moves (not cvt.rn.f32),
// and that negative pointer offsets sign-extend correctly.

// ------------------------------------------------------------------
// Comparison type promotion: comparing UINT32 values must emit setp.u32.
// If the compiler uses s32 comparisons, values >= 0x80000000 will appear
// negative and compare incorrectly with smaller unsigned values.

__global__ void cmp_type_probe(unsigned int *out, unsigned int *a, int n) {
    unsigned int tid = threadIdx.x;
    if (tid < (unsigned int)n) {
        unsigned int v = a[tid];
        unsigned int big = 0xC0000000u;
        // If tid > big (unsigned comparison): these should be false for small tid
        if (v > big) {
            out[tid] = 1u;
        } else {
            out[tid] = 0u;
        }
    }
}

// ------------------------------------------------------------------
// Mixed signed/unsigned comparison: comparing INT32 with UINT32.
// Result type should use unsigned semantics to avoid treating large
// unsigned values as negative.

__global__ void mixed_cmp_probe(int *out, unsigned int *a, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = a[tid];
        unsigned int limit = 0x80000000u;
        // If v >= limit (unsigned): 2^31 and above should pass
        if (v >= limit) {
            out[tid] = 1;
        } else {
            out[tid] = 0;
        }
    }
}

// ------------------------------------------------------------------
// Warp shuffle with float: __shfl_sync must preserve float bits.
// A naive implementation might emit cvt.rn.f32.s32 (numeric conversion)
// instead of just using the b32 register directly.

__global__ void shfl_float_probe(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        // Shuffle down by 1 — lane 0 gets lane 1's value
        float shuffled = __shfl_down_sync(0xFFFFFFFF, v, 1);
        out[tid] = shuffled;
    }
}

// ------------------------------------------------------------------
// Negative pointer index: a[tid - offset] where the result can be negative.
// The subtraction should use cvt.s64.s32 (sign-extend) for the byte offset
// so that negative indices produce large negative 64-bit offsets that wrap.

__global__ void neg_index_probe(float *out, float *data, int n) {
    int tid = threadIdx.x;
    // Only process threads where tid >= 2 to avoid OOB in correctness testing,
    // but the compiler must use signed widening regardless
    if (tid >= 2 && tid < n) {
        // tid - 2: for signed INT32 tid, this is a signed subtraction
        out[tid] = data[tid - 2];
    }
}
