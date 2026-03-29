// Probe: device function calls inside loops,
// device function with loop inside it called from a loop,
// ternary expression used as statement (result discarded vs used),
// compound conditions in for-loop with function call

// Helper: clamp to [lo, hi]
__device__ int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Helper: integer abs
__device__ int iabs(int v) {
    return v < 0 ? -v : v;
}

// Call device function inside for-loop body
__global__ void clamp_array(int *out, int *in, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clampi(in[tid], lo, hi);
    }
}

// Call device function inside for-loop with accumulator
__global__ void sum_abs(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += iabs(in[i]);
        }
        *out = total;
    }
}

// Device function with an internal loop, called from a loop
__device__ int count_bits(int v) {
    int cnt = 0;
    for (int b = 0; b < 32; b++) {
        if (v & (1 << b)) cnt++;
    }
    return cnt;
}

__global__ void popcount_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = count_bits(in[tid]);
    }
}

// Ternary used in accumulation
__global__ void ternary_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos = 0, neg = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            // ternary selects which counter to increment
            pos += (v >= 0) ? 1 : 0;
            neg += (v < 0)  ? 1 : 0;
        }
        out[0] = pos;
        out[1] = neg;
    }
}
