// Probe: advanced memory patterns — atomic on shared memory structs,
// warp-level memory consistency, aliased pointer writes, and
// last few untested intrinsics.

// ------------------------------------------------------------------
// __ldg on struct fields via pointer arithmetic.

struct Pair32 { int first; int second; };

__global__ void ldg_struct_field(int *out, struct Pair32 *pairs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Load individual fields via __ldg on int pointers
        int *base = (int *)pairs + tid * 2;
        int a = __ldg(base + 0);   // first field
        int b = __ldg(base + 1);   // second field
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// __noinline__ device fn (qualifier should be consumed/ignored).

__device__ __noinline__ int heavy_fn(int x) {
    int r = x;
    for (int i = 0; i < 4; i++) {
        r = (r * 1664525 + 1013904223);  // LCG
    }
    return r;
}

__global__ void noinline_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = heavy_fn(in[tid]);
    }
}

// ------------------------------------------------------------------
// atomicCAS on 64-bit (long long).

__global__ void atomic_cas_ll(long long *out, long long *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        long long old = in[tid];
        long long expected = old;
        long long new_val  = old + 1LL;
        // atomicCAS returns old value; compare to see if swap happened
        long long result = atomicCAS((unsigned long long *)out,
                                     (unsigned long long)expected,
                                     (unsigned long long)new_val);
        // Store whether swap succeeded
        in[tid] = (result == expected) ? 1LL : 0LL;
    }
}

// ------------------------------------------------------------------
// __float2uint_rn, __uint2float_rn.

__global__ void uint_float_cvt(unsigned int *out_u, float *out_f,
                                 float *in_f, unsigned int *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_u[tid] = __float2uint_rn(in_f[tid]);
        out_f[tid] = __uint2float_rn(in_u[tid]);
    }
}

// ------------------------------------------------------------------
// __ll2float_rn on long long input.

__global__ void ll_to_float(float *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ll2float_rn(in[tid]);
    }
}

// ------------------------------------------------------------------
// Warp-level compare: find minimum via shfl (int version).

__global__ void warp_min_int(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = (gid < n) ? in[gid] : 2147483647;

    v = min(v, __shfl_xor_sync(0xFFFFFFFF, v, 16));
    v = min(v, __shfl_xor_sync(0xFFFFFFFF, v,  8));
    v = min(v, __shfl_xor_sync(0xFFFFFFFF, v,  4));
    v = min(v, __shfl_xor_sync(0xFFFFFFFF, v,  2));
    v = min(v, __shfl_xor_sync(0xFFFFFFFF, v,  1));

    if ((threadIdx.x & 31) == 0 && gid < n) {
        out[gid / 32] = v;
    }
}

// ------------------------------------------------------------------
// Multiple __syncthreads() in one kernel.

__global__ void multi_sync(int *out, int *in, int n) {
    __shared__ int s1[128], s2[128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    s1[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    s2[tid] = s1[(tid + 1) % 128];
    __syncthreads();

    if (gid < n) {
        out[gid] = s1[tid] + s2[tid];
    }
}

// ------------------------------------------------------------------
// Texture fetch simulation via __ldg with offset.

__global__ void texfetch_sim(float *out, float *tex, float *coords,
                               int tex_w, int tex_h, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fx = coords[tid * 2 + 0];
        float fy = coords[tid * 2 + 1];
        int x = (int)fx & (tex_w - 1);  // wrap
        int y = (int)fy & (tex_h - 1);
        out[tid] = __ldg(&tex[y * tex_w + x]);
    }
}

// ------------------------------------------------------------------
// __forceinline__ device fn (qualifier consumed, fn inlined normally).

__device__ __forceinline__ float fast_normalize(float x, float y) {
    float len = sqrtf(x*x + y*y);
    return (len > 0.0f) ? x / len : 0.0f;
}

__global__ void forceinline_kernel(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fast_normalize(xs[tid], ys[tid]);
    }
}
