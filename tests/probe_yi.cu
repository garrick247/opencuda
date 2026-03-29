// Probe: printf in kernel (vprintf), __device__ array global, multiple
// struct types in same translation unit, struct with padding (alignment),
// for-loop with continue inside switch inside loop, cast to/from void*,
// size_t arithmetic, conditional initialization patterns, and
// __device__ function that reads __constant__ memory.

// ------------------------------------------------------------------
// printf in kernel (single format, single arg).

__global__ void printf_kernel(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && in[tid] < 0) {
        printf("negative at %d\n", tid);
    }
}

// ------------------------------------------------------------------
// __device__ global array (read by all threads).

__device__ int g_lookup[16] = {
    0, 1, 1, 2, 1, 2, 2, 3,
    1, 2, 2, 3, 2, 3, 3, 4
};

__global__ void device_arr_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 0xF;
        out[tid] = g_lookup[v];
    }
}

// ------------------------------------------------------------------
// Multiple struct types in same TU (struct X uses struct Y).

struct Interval { float lo, hi; };
struct BBox2 { struct Interval x, y; };

__device__ int bbox_contains(struct BBox2 b, float px, float py) {
    return (px >= b.x.lo && px <= b.x.hi &&
            py >= b.y.lo && py <= b.y.hi) ? 1 : 0;
}

__global__ void bbox_kernel(int *out, float *px, float *py,
                               float xlo, float xhi, float ylo, float yhi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct BBox2 b;
        b.x.lo = xlo; b.x.hi = xhi;
        b.y.lo = ylo; b.y.hi = yhi;
        out[tid] = bbox_contains(b, px[tid], py[tid]);
    }
}

// ------------------------------------------------------------------
// for-loop with continue inside switch inside loop.

__global__ void switch_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i] & 3;
            switch (v) {
                case 0: continue;         // skip this element
                case 1: s += 1; break;
                case 2: s += 4; break;
                case 3: s += 9; break;
            }
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Conditional initialization: var set in all paths before use.

__global__ void cond_init(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int r;   // declared but not yet initialized
        if (x > 0)      r = x * 2;
        else if (x < 0) r = -x;
        else             r = 0;
        // r is now defined in all paths
        out[tid] = r + b[tid];
    }
}

// ------------------------------------------------------------------
// size_t arithmetic.

__global__ void sizet_arith(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        size_t s = (size_t)tid * sizeof(int);
        out[tid] = (int)(s / sizeof(int));  // should equal tid
    }
}

// ------------------------------------------------------------------
// __device__ function that reads __constant__ memory.

__constant__ float c_scale[4] = {1.0f, 2.0f, 4.0f, 8.0f};

__device__ float scale_by_idx(float v, int idx) {
    return v * c_scale[idx & 3];
}

__global__ void const_dev_kernel(float *out, float *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = scale_by_idx(in[tid], idx[tid]);
}

// ------------------------------------------------------------------
// Cast to/from void* (used for generic data moves).

__global__ void void_ptr_cast(int *out, void *src, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *typed = (int *)src;
        out[tid] = typed[tid] + 1;
    }
}

// ------------------------------------------------------------------
// Accumulate into output that is also read (read-modify-write through local).

__global__ void rmw_local(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = out[tid];  // read current value
        for (int i = 0; i < 4; i++) acc += in[(tid + i) % n];
        out[tid] = acc;       // write updated value
    }
}
