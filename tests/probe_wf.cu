// Probe: int3/uint3/dim3 built-in vectors, warpSize constant, umin/umax,
// __match_all_sync, nested struct copy, and conditional assignment chains.

// ------------------------------------------------------------------
// int3 / uint3 built-in vector types.

__global__ void int3_ops(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int3 v;
        v.x = tid;
        v.y = tid + 1;
        v.z = tid + 2;
        out[tid] = v.x + v.y + v.z;  // 3*tid + 3
    }
}

__global__ void uint3_ops(unsigned int *out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    if (tid < n) {
        uint3 u;
        u.x = tid * 2;
        u.y = tid * 3;
        u.z = tid * 5;
        out[tid] = u.x + u.y + u.z;  // 10*tid
    }
}

// ------------------------------------------------------------------
// blockDim.x/y/z product.

__global__ void blockdim_product(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int bx = blockDim.x;
        int by = blockDim.y;
        int bz = blockDim.z;
        out[tid] = bx * by * bz;
    }
}

// ------------------------------------------------------------------
// warpSize constant (always 32 on NVIDIA).

__global__ void warpsize_kernel(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = tid % warpSize;  // lane ID
    }
}

// ------------------------------------------------------------------
// min/max with unsigned types (umin/umax in PTX).

__global__ void umin_umax(unsigned int *out, unsigned int *in, unsigned int n) {
    unsigned int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int lo = min(v, 100u);
        unsigned int hi = max(lo, 0u);
        out[tid] = hi;
    }
}

// ------------------------------------------------------------------
// __match_all_sync: all lanes have the same value.

__global__ void match_all_sync_kernel(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int mask_out;
        int all_same = __match_all_sync(0xFFFFFFFF, v, &mask_out);
        out[gid] = all_same;
    }
}

// ------------------------------------------------------------------
// Nested struct copy (copy whole inner struct field).

struct Inner { int a; float b; };
struct Outer { int id; struct Inner data; };

__global__ void nested_struct_copy(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Outer src;
        src.id = tid;
        src.data.a = in[tid];
        src.data.b = (float)in[tid] * 1.5f;

        struct Outer dst;
        dst = src;   // whole-struct copy

        out[tid] = dst.data.b + (float)dst.data.a;  // 2.5 * in[tid]
    }
}

// ------------------------------------------------------------------
// Conditional assignment chains (sequence of conditional ternaries).

__global__ void cond_assign_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = (v > 0) ? v : 0;
        int b = (a > 10) ? a - 10 : a;
        int c = (b > 5) ? b * 2 : b;
        out[tid] = c;
    }
}

// ------------------------------------------------------------------
// Unsigned right shift (logical, not arithmetic).

__global__ void unsigned_rshift(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = v >> 4;   // logical right shift (no sign extension)
    }
}

// ------------------------------------------------------------------
// Signed right shift (arithmetic shift: sign-extends).

__global__ void signed_rshift(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = v >> 4;   // arithmetic right shift
    }
}

// ------------------------------------------------------------------
// Left shift with large count.

__global__ void large_shift(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = (v & 1) << 31;   // shift to sign bit
        out[tid] = r;
    }
}
