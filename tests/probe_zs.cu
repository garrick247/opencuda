// Probe: really push the limits — 1000-element local array, very deep
// nesting (8 if levels), switch inside for inside if, massive shared mem
// (48KB), long expression chain (20 ops), and stress all atomic types.

// ------------------------------------------------------------------
// Large local array (256 elements).

__global__ void large_local(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int buf[256];
    // Initialize
    for (int i = 0; i < 256; i++) buf[i] = in[(tid + i) % n];
    // Scan
    for (int i = 1; i < 256; i++) buf[i] += buf[i-1];
    out[tid] = buf[255];
}

// ------------------------------------------------------------------
// 8-level deep nesting.

__global__ void deep8(int *out, int *a, int *b, int *c, int *d,
                         int *e, int *f, int *g, int *h, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = 0;
        if (a[tid] > 0) {
            if (b[tid] > 0) {
                if (c[tid] > 0) {
                    if (d[tid] > 0) {
                        if (e[tid] > 0) {
                            if (f[tid] > 0) {
                                if (g[tid] > 0) {
                                    if (h[tid] > 0) {
                                        r = 255;
                                    } else r = 128;
                                } else r = 64;
                            } else r = 32;
                        } else r = 16;
                    } else r = 8;
                } else r = 4;
            } else r = 2;
        } else r = 1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Switch inside for inside if (triple nesting).

__global__ void triple_nest(int *out, int *ops, int *vals, int n_ops, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < n_ops; i++) {
            int op = ops[i];
            int v = vals[i];
            switch (op) {
                case 0: acc += v; break;
                case 1: acc -= v; break;
                case 2: acc *= v; break;
                case 3: if (v != 0) acc /= v; break;
                case 4: acc &= v; break;
                case 5: acc |= v; break;
                case 6: acc ^= v; break;
                default: break;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Stress all integer atomic types: int, unsigned, long long, float, double.

__global__ void atomic_all_types(int *out_i, unsigned *out_u,
                                    long long *out_ll, float *out_f,
                                    double *out_d,
                                    int *in_i, unsigned *in_u,
                                    long long *in_ll, float *in_f,
                                    double *in_d, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAdd(out_i, in_i[gid]);
        atomicAdd(out_u, in_u[gid]);
        atomicAdd(out_ll, in_ll[gid]);
        atomicAdd(out_f, in_f[gid]);
        atomicAdd(out_d, in_d[gid]);
    }
}

// ------------------------------------------------------------------
// 20-op expression chain.

__global__ void long_expr(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float x = in[gid];
    float r = x;
    r = r * x + 1.0f;
    r = r * x - 2.0f;
    r = r * x + 3.0f;
    r = r * x - 4.0f;
    r = r + x * x;
    r = r - x * 0.5f;
    r = r * (x + 1.0f);
    r = r / (fabsf(x) + 1.0f);
    r = r + sqrtf(fabsf(r));
    r = r * __expf(-fabsf(r) * 0.01f);
    r = r + __sinf(x * 0.1f);
    r = r - __cosf(x * 0.2f);
    r = r * fmaxf(r, 0.0f);
    r = r + fminf(x, r);
    r = r * r * 0.001f;
    r = r + 42.0f;
    r = r - (float)(int)r;  // fractional part
    r = r * 1000.0f;
    r = floorf(r);
    out[gid] = r;
}

// ------------------------------------------------------------------
// Maximum shared memory usage (48KB / 4 bytes = 12288 floats).

__global__ void max_shared(float *out, float *in, int n) {
    __shared__ float big[12288];
    int tid = threadIdx.x;
    // Fill shared memory
    for (int i = tid; i < 12288; i += blockDim.x) {
        big[i] = (i < n) ? in[i] : 0.0f;
    }
    __syncthreads();
    // Read back a few values
    if (tid < n) {
        float s = 0.0f;
        for (int k = 0; k < 8; k++) {
            int idx = (tid * 7 + k) % 12288;
            s += big[idx];
        }
        out[tid] = s;
    }
}
