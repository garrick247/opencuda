// Probe: narrowing stores (int→short, int→char), unsigned 64-bit comparison,
// __syncthreads_count/__syncthreads_and/__syncthreads_or,
// const pointer to struct, __ldlu/__ldcv/__ldca load qualifiers,
// bitfield struct read/write, string literal in printf,
// conditional with function-call side effect, and warp prefix-min.

// ------------------------------------------------------------------
// Narrowing stores: int → short* and int → char* (truncating stores).

__global__ void narrow_store_short(short *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = (short)v;   // truncate to 16 bits
    }
}

__global__ void narrow_store_char(signed char *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (signed char)in[tid];   // truncate to 8 bits
    }
}

// ------------------------------------------------------------------
// Unsigned 64-bit comparison.

__global__ void ull_cmp(int *out, unsigned long long *a, unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long x = a[tid];
        unsigned long long y = b[tid];
        out[tid] = (x < y) ? -1 : (x > y) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// __syncthreads_count / __syncthreads_and / __syncthreads_or.

__global__ void syncthreads_ops(int *out_count, int *out_and, int *out_or,
                                   int *in, int n) {
    int tid = threadIdx.x;
    int v = (tid < n) ? in[tid] : 0;
    int cnt = __syncthreads_count(v > 0);
    int and_ = __syncthreads_and(v != 0);
    int or_  = __syncthreads_or(v > 0);
    if (tid < n) {
        out_count[tid] = cnt;
        out_and[tid]   = and_;
        out_or[tid]    = or_;
    }
}

// ------------------------------------------------------------------
// const pointer to struct (read-only struct access).

struct Params { float alpha, beta; int n; };

__device__ float apply_params(const struct Params *p, float x) {
    return p->alpha * x + p->beta;
}

__global__ void const_ptr_struct(float *out, float *in,
                                    const struct Params *params, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = apply_params(params, in[tid]);
}

// ------------------------------------------------------------------
// __ldlu / __ldcv (load with cache-control qualifiers).

__global__ void ldlu_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = __ldlu(in + tid);   // load last-use
        out[tid] = v + 1;
    }
}

__global__ void ldcv_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = __ldcv(in + tid);  // cache-volatile load
        out[tid] = v * 2.0f;
    }
}

// ------------------------------------------------------------------
// printf with multiple format args.

__global__ void printf_multi(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && in[tid] == 42) {
        printf("thread %d found value %d\n", tid, in[tid]);
    }
}

// ------------------------------------------------------------------
// Conditional with function-call side-effect as condition.

__device__ int side_effect_fn(int *counter, int x) {
    (*counter)++;
    return x > 0;
}

__global__ void cond_side_effect(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int cnt = 0;
        int v = in[tid];
        int r = side_effect_fn(&cnt, v) ? v * 2 : -v;
        out[tid] = r + cnt;
    }
}

// ------------------------------------------------------------------
// Warp prefix minimum using shfl_down_sync.

__global__ void warp_prefix_min(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = (gid < n) ? in[gid] : 0x7FFFFFFF;
    // Reduce min across warp
    int t;
    t = __shfl_down_sync(0xFFFFFFFF, v, 16); if (t < v) v = t;
    t = __shfl_down_sync(0xFFFFFFFF, v,  8); if (t < v) v = t;
    t = __shfl_down_sync(0xFFFFFFFF, v,  4); if (t < v) v = t;
    t = __shfl_down_sync(0xFFFFFFFF, v,  2); if (t < v) v = t;
    t = __shfl_down_sync(0xFFFFFFFF, v,  1); if (t < v) v = t;
    if (lane == 0 && gid < n) out[gid / 32] = v;
}

// ------------------------------------------------------------------
// Float infinity and large constant handling.

__global__ void inf_constant(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Clamp to ±1e30 (large but finite)
        float clamped = (v > 1e30f) ? 1e30f : (v < -1e30f) ? -1e30f : v;
        out[tid] = clamped;
    }
}
