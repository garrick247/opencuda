// Probe: __restrict__ qualifier, for-loop with compound initializer,
// complex LICM with address computation, and labeled-break-like patterns.

// ------------------------------------------------------------------
// __restrict__ on both parameters — should parse; semantics not used.

__global__ void restrict_ptrs(float * __restrict__ out,
                               const float * __restrict__ in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f + in[tid] * in[tid];
    }
}

// ------------------------------------------------------------------
// For loop with multiple variables in init (C99 style — one type only).

__global__ void for_multi_init(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        // Multi-variable init: for(int i = 0, j = 8; i < 4; i++, j--)
        for (int i = 0, j = 8; i < 4; i++, j--) {
            acc += v * i + j;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// LICM: address computation with loop-invariant base and loop-carried offset.

__global__ void licm_addr(float *out, float *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        // base address is loop-invariant
        float *row = in + tid * stride;
        float acc = 0.0f;
        for (int i = 0; i < stride && i < 16; i++) {
            acc += row[i];  // row is invariant, i is loop-carried
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Switch with return inside case — complex exit paths.

__device__ int score_level(int level, int score) {
    switch (level) {
        case 0: return score;
        case 1: return score * 2;
        case 2: return score * 3 + 10;
        case 3: return score * 5 + 25;
        default: return -1;
    }
}

__global__ void switch_return(int *out, int *levels, int *scores, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = score_level(levels[tid], scores[tid]);
    }
}

// ------------------------------------------------------------------
// Break from nested loop affecting outer accumulator.

__global__ void break_outer_acc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int found = 0;
        for (int i = 0; i < 8 && !found; i++) {
            for (int j = 0; j < 8; j++) {
                int val = i * 8 + j;
                if (val == v) {
                    acc = val;
                    found = 1;
                    break;
                }
                acc += 1;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// LICM: invariant function call result used in loop.

__device__ float compute_scale(float a, float b) {
    return a / (b + 1.0f);
}

__global__ void licm_call_result(float *out, float *in, float *params, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // scale is computed from loop-invariant params — should be hoisted
        float scale = compute_scale(params[0], params[1]);
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += v * scale * (float)i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Goto-free "labeled break" simulation: flag + outer loop exit.

__global__ void flag_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        int stop = 0;
        for (int i = 0; i < 4 && !stop; i++) {
            for (int j = 0; j < 4 && !stop; j++) {
                result += i + j;
                if (result > v) stop = 1;
            }
        }
        out[tid] = result;
    }
}
