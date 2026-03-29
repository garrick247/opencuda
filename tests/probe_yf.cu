// Probe: SSA/CFG stress — return inside nested if-else, return inside loop,
// infinite loop (no natural exit — only break), unreachable-after-return,
// variable declared in one arm of if only (C90 restriction we should handle),
// loop with update in multiple places, deeply-nested if (5 levels),
// and mixed int/float in the same expression chain.

// ------------------------------------------------------------------
// Return inside nested if-else (early multi-path return).

__device__ int rank_nested(int a, int b, int c) {
    if (a > b) {
        if (a > c) return 1;   // a is max
        else       return 2;   // c is max
    } else {
        if (b > c) return 3;   // b is max
        else       return 4;   // c is max
    }
}

__global__ void rank_nested_kernel(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = rank_nested(a[tid], b[tid], c[tid]);
}

// ------------------------------------------------------------------
// Return inside a loop.

__device__ int first_positive(int *arr, int len) {
    for (int i = 0; i < len; i++) {
        if (arr[i] > 0) return arr[i];
    }
    return -1;
}

__global__ void first_pos_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Build a small per-thread window and scan it
        int window[4];
        for (int k = 0; k < 4; k++) window[k] = in[(tid + k) % n];
        out[tid] = first_positive(window, 4);
    }
}

// ------------------------------------------------------------------
// Loop with no natural exit (only break).

__global__ void break_only_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0;
        while (1) {
            if (v <= 1 || i >= 32) break;
            if (v % 2 == 0) v /= 2;
            else v = 3 * v + 1;
            i++;
        }
        out[tid] = i;
    }
}

// ------------------------------------------------------------------
// Deeply nested if (5 levels).

__device__ int deep_if(int a, int b, int c, int d, int e) {
    if (a > 0) {
        if (b > 0) {
            if (c > 0) {
                if (d > 0) {
                    if (e > 0) return 1;
                    else       return 2;
                } else         return 3;
            } else             return 4;
        } else                 return 5;
    } else                     return 0;
}

__global__ void deep_if_kernel(int *out, int *a, int *b, int *c,
                                  int *d, int *e, int n) {
    int tid = threadIdx.x;
    if (tid < n)
        out[tid] = deep_if(a[tid], b[tid], c[tid], d[tid], e[tid]);
}

// ------------------------------------------------------------------
// Loop with variable updated at multiple points (complex phi).

__global__ void multi_update_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0, acc = 1;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) {
                s += v;
                acc *= 2;
            } else if (v < 0) {
                s -= v;
                acc++;
            }
            // else: neither branch updates s/acc
        }
        out[tid] = s + acc;
    }
}

// ------------------------------------------------------------------
// Mixed int/float in same expression chain.

__global__ void mixed_chain(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        float f = (float)x * 1.5f + 0.5f;
        int   i = (int)f;
        float g = f - (float)i;   // fractional part
        out[tid] = g + (float)(i % 7);
    }
}

// ------------------------------------------------------------------
// Conditional write through pointer (branch with pointer store).

__global__ void cond_ptr_store(float *out, float *in, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float *dst = &out[tid];
        if (v > threshold) {
            *dst = v * v;
        } else {
            *dst = -v;
        }
    }
}

// ------------------------------------------------------------------
// String-free printf substitute: write formatted value into int array.
// (Tests multi-arg expression with division and modulo.)

__global__ void format_digits(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v < 0) v = -v;
        int d0 = v % 10;
        int d1 = (v / 10) % 10;
        int d2 = (v / 100) % 10;
        out[tid * 3    ] = d2;
        out[tid * 3 + 1] = d1;
        out[tid * 3 + 2] = d0;
    }
}

// ------------------------------------------------------------------
// Kernel with __restrict__ on multiple pointer params.

__global__ void restrict_kernel(float * __restrict__ out,
                                  const float * __restrict__ a,
                                  const float * __restrict__ b,
                                  int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = a[tid] + b[tid];
}
