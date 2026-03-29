// Probe: memory fence variants, async patterns, vectorized loads (int2/int4/float4),
// ldg on various types, and memory-ordering patterns.

// ------------------------------------------------------------------
// int2 struct load/store via global memory.

__global__ void int2_ops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 2];
        int b = in[tid * 2 + 1];
        // Simulate int2 pack/unpack
        out[tid * 2]     = a + b;
        out[tid * 2 + 1] = a - b;
    }
}

// ------------------------------------------------------------------
// float4: 4 consecutive floats treated as a group.

__global__ void float4_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid * 4 + 0];
        float y = in[tid * 4 + 1];
        float z = in[tid * 4 + 2];
        float w = in[tid * 4 + 3];
        float dot = x*x + y*y + z*z + w*w;
        out[tid] = dot;
    }
}

// ------------------------------------------------------------------
// Fused multiply-add: __fmaf_rn.

__global__ void fmaf_kernel(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __fmaf_rn(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// __fma_rn for double.

__global__ void fma_double(double *out, double *a, double *b, double *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __fma_rn(a[tid], b[tid], c[tid]);
    }
}

// ------------------------------------------------------------------
// __threadfence() — device-level memory fence (not system).

__global__ void threadfence_pattern(int *flag, int *data, int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Producer: write data then flag
        data[0] = 42;
        __threadfence();
        flag[0] = 1;
    }
    __syncthreads();
    if (tid < n && flag[0] == 1) {
        out[tid] = data[0];
    }
}

// ------------------------------------------------------------------
// Bitfield extraction using masks and shifts.

__global__ void bitfield_extract(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int lo4  = (v >>  0) & 0xF;
        unsigned int hi4  = (v >> 28) & 0xF;
        unsigned int mid8 = (v >> 12) & 0xFF;
        out[tid] = (lo4 << 12) | (hi4 << 8) | mid8;
    }
}

// ------------------------------------------------------------------
// __clz (count leading zeros).

__global__ void clz_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __clz(in[tid]);
    }
}

// ------------------------------------------------------------------
// __popc (popcount).

__global__ void popc_kernel(int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __popc(in[tid]);
    }
}

// ------------------------------------------------------------------
// __brev (bit reverse).

__global__ void brev_kernel(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __brev(in[tid]);
    }
}

// ------------------------------------------------------------------
// Warp-level exclusive prefix sum using shfl.

__global__ void warp_prefix_sum(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int lane = threadIdx.x & 31;

        // Inclusive scan via shfl_up
        for (int offset = 1; offset < 32; offset <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFF, v, offset);
            if (lane >= offset) v += y;
        }
        // Convert to exclusive
        out[gid] = v - in[gid];
    }
}

// ------------------------------------------------------------------
// Min/max tree reduction via shfl_xor.

__global__ void warp_min_reduce(int *out, int *in, int n) {
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
