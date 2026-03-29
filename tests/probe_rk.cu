// Probe: warpSize, __mul24/__umul24, __mulhi/__umulhi, __hadd/__rhadd,
// __byte_perm, stride loops with gridDim/blockDim, and __funnelshift.

// ------------------------------------------------------------------
// warpSize: special register (should be %warpsize or constant 32).

__global__ void warpsize_kernel(int *out, int n) {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    if (tid < n) {
        out[tid] = lane;
    }
}

// ------------------------------------------------------------------
// __mul24 / __umul24: 24-bit integer multiply (low 32 bits of product).

__global__ void mul24_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __mul24(a[tid], b[tid]);
    }
}

__global__ void umul24_kernel(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __umul24(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __mulhi / __umulhi: high 32 bits of 32x32 multiply.

__global__ void mulhi_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __mulhi(a[tid], b[tid]);
    }
}

__global__ void umulhi_kernel(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __umulhi(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __hadd / __rhadd: halving add (no overflow): (a+b)/2.

__global__ void hadd_kernel(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __hadd(a[tid], b[tid]);
    }
}

__global__ void rhadd_kernel(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __rhadd(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// __byte_perm: byte permutation from two 32-bit words.

__global__ void byte_perm_kernel(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Select bytes 0,1,2,3 from a (selector 0x3210)
        out[tid] = __byte_perm(a[tid], b[tid], 0x3210u);
    }
}

// ------------------------------------------------------------------
// Stride loop using gridDim and blockDim.

__global__ void stride_loop(float *out, float *in, int n) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += stride) {
        out[i] = in[i] * 2.0f;
    }
}

// ------------------------------------------------------------------
// __funnelshift_l / __funnelshift_r: funnel shift.

__global__ void funnelshift_kernel(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[2*tid]   = __funnelshift_l(a[tid], b[tid], 8);
        out[2*tid+1] = __funnelshift_r(a[tid], b[tid], 8);
    }
}
