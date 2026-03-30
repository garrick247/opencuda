// Probe: optimizer/codegen interaction stress — dead code after return in
// device function, unreachable else-branch after early return, loop with
// break that creates dead phi, complex address computation in array of struct,
// multiple atomicCAS in spin-lock pattern, warp divergent branch where
// half the warp returns early, __float2half_rn on computed expression,
// and store-to-different-output-arrays inside same branch.

// ------------------------------------------------------------------
// Dead code after return: compiler must not emit instructions after ret.

__device__ int dead_after_return(int x) {
    if (x > 0) return x * 2;
    return -x;
    // Dead code — should be ignored
    int y = x + 10;
    return y;
}

__global__ void dead_return_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = dead_after_return(in[tid]);
}

// ------------------------------------------------------------------
// Loop with break that creates dead phi at merge point.

__global__ void break_dead_phi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int last = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v < 0) {
                last = -1;
                break;
            }
            s += v;
            last = v;
        }
        // Both s and last are live after loop; phi merges break and natural exit
        out[tid] = s + last;
    }
}

// ------------------------------------------------------------------
// Complex address computation: array of struct with computed field offset.

struct Pixel { unsigned char r, g, b, a; };

__global__ void pixel_extract(int *out_r, int *out_g, int *out_b,
                                struct Pixel *px, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Pixel p = px[tid];
        out_r[tid] = (int)p.r;
        out_g[tid] = (int)p.g;
        out_b[tid] = (int)p.b;
    }
}

// ------------------------------------------------------------------
// Multiple atomicCAS to implement spin-lock (single-slot pattern).

__global__ void spin_lock_sim(int *lock, int *data, int val, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // Try atomicCAS once (not actually spinning — just testing emission)
        int old = atomicCAS(lock, 0, 1);
        if (old == 0) {
            // Won the lock
            data[gid] = val;
            atomicExch(lock, 0);  // release
        }
    }
}

// ------------------------------------------------------------------
// Warp divergence: half the warp returns early.

__global__ void half_warp_return(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    if (gid >= n) return;
    float v = in[gid];
    // Even lanes return early with identity
    if (lane % 2 == 0) {
        out[gid] = v;
        return;
    }
    // Odd lanes compute more
    v = v * v + 1.0f;
    out[gid] = v;
}

// ------------------------------------------------------------------
// __float2half_rn on a computed expression (not just a load).

__global__ void f2h_computed(unsigned short *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float expr = a[tid] * b[tid] + 0.5f;
        __half h = __float2half_rn(expr);
        out[tid] = __half_as_ushort(h);
    }
}

// ------------------------------------------------------------------
// Store to different output arrays in both branches (phi on pointer).

__global__ void branch_store(int *out_pos, int *out_neg, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v >= 0) {
            out_pos[tid] = v;
            out_neg[tid] = 0;
        } else {
            out_pos[tid] = 0;
            out_neg[tid] = -v;
        }
    }
}

// ------------------------------------------------------------------
// Cascade of device calls with accumulator.

__device__ int step1(int x) { return x + 3; }
__device__ int step2(int x) { return x * 2; }
__device__ int step3(int x) { return x - 1; }

__global__ void cascade_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        v = step1(v);
        v = step2(v);
        v = step3(v);
        out[tid] = v;  // (in+3)*2-1
    }
}

// ------------------------------------------------------------------
// Complex shared-mem reduction with variable block size.

__global__ void flex_reduce_sum(float *out, float *in, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    // Tree reduction over actual blockDim
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ------------------------------------------------------------------
// Compound assignment on pointer dereference: *p += val.

__global__ void deref_compound(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = &out[tid];
        *p = 0;
        for (int i = 0; i < 4; i++) {
            *p += in[(tid + i) % n];
        }
    }
}
