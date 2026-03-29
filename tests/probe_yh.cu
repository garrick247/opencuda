// Probe: warp-level matrix multiply (wmma-free), texture-free 2D stencil,
// reduction with shared+warp two-stage, prefix scan two-stage (warp+block),
// multiple __global__ calls in same kernel (device function reuse),
// post-increment in array subscript inside loop, compound assignment in
// complex expression, and address arithmetic on struct array.

// ------------------------------------------------------------------
// 2D 5-point stencil without texture.

__global__ void stencil5pt(float *out, float *in, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < W-1 && y > 0 && y < H-1) {
        float c  = in[y * W + x];
        float xm = in[y * W + (x-1)];
        float xp = in[y * W + (x+1)];
        float ym = in[(y-1) * W + x];
        float yp = in[(y+1) * W + x];
        out[y * W + x] = 0.2f * (c + xm + xp + ym + yp);
    }
}

// ------------------------------------------------------------------
// 2-stage reduction: warp-level then block-level via shared memory.

__global__ void two_stage_reduce(float *out, float *in, int n) {
    __shared__ float warp_sums[32];
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float v = (gid < n) ? in[gid] : 0.0f;
    // Stage 1: warp reduce
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    // Stage 2: first warp reduces warp_sums
    if (wid == 0) {
        v = (lane < 32) ? warp_sums[lane] : 0.0f;
        v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
        v += __shfl_xor_sync(0xFFFFFFFF, v,  1);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ------------------------------------------------------------------
// 2-stage inclusive prefix scan (warp + block level).

__global__ void two_stage_scan(int *out, int *in, int n) {
    __shared__ int warp_ends[32];
    int gid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    int v = (gid < n) ? in[gid] : 0;
    // Intra-warp scan
    for (int d = 1; d < 32; d <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if (lane >= d) v += t;
    }
    if (lane == 31) warp_ends[wid] = v;
    __syncthreads();
    // Scan warp_ends in first warp
    if (wid == 0) {
        int w = (lane < 32) ? warp_ends[lane] : 0;
        for (int d = 1; d < 32; d <<= 1) {
            int t = __shfl_up_sync(0xFFFFFFFF, w, d);
            if (lane >= d) w += t;
        }
        warp_ends[lane] = w;
    }
    __syncthreads();
    // Add warp offset
    if (wid > 0) v += warp_ends[wid - 1];
    if (gid < n) out[gid] = v;
}

// ------------------------------------------------------------------
// Device function reused multiple times from same kernel.

__device__ float poly2(float x, float a, float b, float c) {
    return a * x * x + b * x + c;
}

__global__ void multi_poly_kernel(float *out, float *x, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float xv = x[tid];
        // Call poly2 three times with different coefficients
        float p1 = poly2(xv, 1.0f, -2.0f, 1.0f);   // (x-1)^2
        float p2 = poly2(xv, 2.0f,  0.0f, -1.0f);  // 2x^2 - 1
        float p3 = poly2(xv, -1.0f, 3.0f, 0.0f);   // -x^2 + 3x
        out[tid] = p1 + p2 + p3;
    }
}

// ------------------------------------------------------------------
// Post-increment inside array subscript inside loop.

__global__ void post_inc_subscript(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int i = 0;
        while (i < 8) {
            s += in[i++];
        }
        out[tid] = s + in[tid % n];
    }
}

// ------------------------------------------------------------------
// Compound assignment in complex expression.

__global__ void compound_complex(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        x += y * 2;         // compound assign
        y -= x / 3;         // compound assign
        x ^= (y << 1);      // compound assign XOR
        out[tid] = x + y;
    }
}

// ------------------------------------------------------------------
// Address arithmetic on array of structs.

struct KeyVal { int key; float val; };

__global__ void aos_arith(int *out_key, float *out_val,
                             struct KeyVal *kv, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Access via pointer arithmetic on struct array
        struct KeyVal *p = kv + tid;
        out_key[tid] = p->key;
        out_val[tid] = p->val;
    }
}

// ------------------------------------------------------------------
// Histogramming with atomicAdd into shared + global.

__global__ void histogram(int *global_hist, int *in, int bins, int n) {
    __shared__ int shared_hist[64];
    int tid = threadIdx.x;
    // Zero shared histogram
    if (tid < bins) shared_hist[tid] = 0;
    __syncthreads();
    // Accumulate
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        int b = in[gid] % bins;
        if (b >= 0) atomicAdd(&shared_hist[b], 1);
    }
    __syncthreads();
    // Merge to global
    if (tid < bins) atomicAdd(&global_hist[tid], shared_hist[tid]);
}
