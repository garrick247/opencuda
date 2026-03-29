// Probe: __ldg on various types, deeply nested if/else phi chains,
// multiple assignment paths, complex loop-carried values, and
// arithmetic on pointer offsets.

// ------------------------------------------------------------------
// __ldg on int, float, double, long long.

__global__ void ldg_types(int *out_i, float *out_f, double *out_d,
                           long long *out_ll,
                           int *in_i, float *in_f, double *in_d,
                           long long *in_ll, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_i[tid]  = __ldg(&in_i[tid]);
        out_f[tid]  = __ldg(&in_f[tid]);
        out_d[tid]  = __ldg(&in_d[tid]);
        out_ll[tid] = __ldg(&in_ll[tid]);
    }
}

// ------------------------------------------------------------------
// Deeply nested if/else: 4 levels → 5 possible values.

__device__ int deep_nest(int v) {
    int r;
    if (v < 0) {
        r = -1;
    } else if (v == 0) {
        r = 0;
    } else if (v < 10) {
        if (v < 5) {
            r = 1;
        } else {
            r = 2;
        }
    } else if (v < 100) {
        r = 3;
    } else {
        r = 4;
    }
    return r;
}

__global__ void deep_nest_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = deep_nest(in[tid]);
    }
}

// ------------------------------------------------------------------
// Loop-carried: accumulate max, min, sum, count simultaneously.

__global__ void quad_accum(float *out_max, float *out_min,
                            float *out_sum, int *out_cnt,
                            float *in, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float mx = in[tid];
        float mn = in[tid];
        float sm = 0.0f;
        int   cnt = 0;

        // Each thread processes strided elements
        for (int i = tid; i < n; i += blockDim.x) {
            float v = in[i];
            if (v > mx) mx = v;
            if (v < mn) mn = v;
            if (v > threshold) {
                sm += v;
                cnt++;
            }
        }
        out_max[tid] = mx;
        out_min[tid] = mn;
        out_sum[tid] = sm;
        out_cnt[tid] = cnt;
    }
}

// ------------------------------------------------------------------
// Chain of dependent assignments: each depends on previous.

__global__ void dep_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v * v;
        int b = a + v;
        int c = b * a - v;
        int d = c ^ b;
        int e = d + (c >> 2);
        int f = e * (a & 0xFF);
        out[tid] = f;
    }
}

// ------------------------------------------------------------------
// __ldg with pointer arithmetic before load.

__global__ void ldg_ptr_arith(float *out, float *in, int *offsets, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *ptr = in + offsets[tid];
        out[tid] = __ldg(ptr);
    }
}

// ------------------------------------------------------------------
// Predicated execution: two writes with different conditions.

__global__ void pred_two_writes(float *a_out, float *b_out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        if (v > 0.0f) {
            a_out[tid] = v * v;
        }
        if (v < 0.0f) {
            b_out[tid] = -v;
        }
        // If v == 0: neither written (undefined output, that's OK for this probe)
    }
}

// ------------------------------------------------------------------
// Sum of absolute differences (SAD) pattern.

__global__ void sad_pattern(int *out, unsigned char *a, unsigned char *b,
                             int n, int block_size) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    if (bid < n && tid < block_size) {
        int idx = bid * block_size + tid;
        int diff = (int)a[idx] - (int)b[idx];
        int absdiff = (diff < 0) ? -diff : diff;
        atomicAdd(&out[bid], absdiff);
    }
}

// ------------------------------------------------------------------
// Packed half-word operations via bit manipulation.

__global__ void packed_halfword(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int lo = v & 0xFFFF;
        unsigned int hi = v >> 16;
        // Swap and increment each halfword
        unsigned int new_lo = (hi + 1) & 0xFFFF;
        unsigned int new_hi = (lo + 1) & 0xFFFF;
        out[tid] = (new_hi << 16) | new_lo;
    }
}

// ------------------------------------------------------------------
// Address-of struct field through global pointer, then store.

struct Config {
    int   max_iter;
    float tol;
    int   n;
};

__global__ void use_config(float *out, float *in, struct Config *cfg) {
    int tid = threadIdx.x;
    int n = cfg->n;
    float tol = cfg->tol;
    int   max_iter = cfg->max_iter;
    if (tid < n) {
        float v = in[tid];
        int iter = 0;
        while (v > tol && iter < max_iter) {
            v = v * 0.5f;
            iter++;
        }
        out[tid] = v;
    }
}
