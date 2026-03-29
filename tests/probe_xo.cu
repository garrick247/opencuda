// Probe: __half2float/__float2half conversions, additional warp intrinsics,
// template-like patterns via macros, and integer types on boundary.

// ------------------------------------------------------------------
// __half2float / __float2half (half-precision convert).

__global__ void half_convert(float *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Convert half (stored as u16) to float
        float v = __half2float(__ushort_as_half(in[tid]));
        out[tid] = v * v;   // square it
    }
}

// ------------------------------------------------------------------
// __float2half_rn and store as u16.

__global__ void float_to_half(unsigned short *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __float2half_rn(in[tid]);
        out[tid] = __half_as_ushort(h);
    }
}

// ------------------------------------------------------------------
// __hadd / __hmul with half precision arithmetic.

__global__ void half_arith(unsigned short *out, unsigned short *a,
                             unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half sum = __hadd(ha, hb);
        __half prod = __hmul(ha, hb);
        // Store sum+prod as half
        out[tid] = __half_as_ushort(__hadd(sum, prod));
    }
}

// ------------------------------------------------------------------
// signed char arithmetic (int8 simulation).

__global__ void int8_arith(signed char *out, signed char *a, signed char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Use int for arithmetic, clamp back to int8 range
        int av = (int)a[tid];
        int bv = (int)b[tid];
        int sum = av + bv;
        // Saturate to [-128, 127]
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        out[tid] = (signed char)sum;
    }
}

// ------------------------------------------------------------------
// unsigned short max reduction.

__global__ void ushort_max_reduce(unsigned short *out, unsigned short *in, int n) {
    __shared__ unsigned short smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (smem[tid + stride] > smem[tid])
                smem[tid] = smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Multi-word atomic: update two global values atomically via spinlock.

__global__ void multi_word_update(int *lock, int *a, int *b, int da, int db, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Spinlock-based critical section
        while (atomicCAS(lock, 0, 1) != 0) {}  // acquire
        *a += da;
        *b += db;
        atomicExch(lock, 0);  // release
    }
}

// ------------------------------------------------------------------
// __isnan / __isinf on doubles.

__global__ void double_special(int *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double v = in[tid];
        int is_nan = __isnan(v);
        int is_inf = __isinf(v);
        out[tid] = is_nan * 2 + is_inf;
    }
}

// ------------------------------------------------------------------
// Complex number multiplication (as separate real/imag arrays).

__device__ void cmul(float *re, float *im, float ar, float ai, float br, float bi) {
    *re = ar * br - ai * bi;
    *im = ar * bi + ai * br;
}

__global__ void complex_mul(float *re_out, float *im_out,
                              float *re_a, float *im_a,
                              float *re_b, float *im_b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float r, i;
        cmul(&r, &i, re_a[tid], im_a[tid], re_b[tid], im_b[tid]);
        re_out[tid] = r;
        im_out[tid] = i;
    }
}

// ------------------------------------------------------------------
// Prefix min via __shfl.

__global__ void shfl_prefix_min(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int v = in[gid];
        int lane = threadIdx.x & 31;

        for (int offset = 1; offset < 32; offset <<= 1) {
            int y = __shfl_up_sync(0xFFFFFFFF, v, offset);
            if (lane >= offset) v = min(v, y);
        }
        out[gid] = v;
    }
}
