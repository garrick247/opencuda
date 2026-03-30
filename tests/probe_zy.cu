// Probe: final frontier — patterns combining every feature —
// recursive device func + struct + shared mem + warp shuffle + atomics,
// kernel calling multiple recursive and non-recursive device functions,
// maximum complexity single kernel.

// ------------------------------------------------------------------
// Recursive tree node with struct.

struct TreeNode { int value; int left; int right; };

__device__ int tree_sum(struct TreeNode *nodes, int idx) {
    if (idx < 0) return 0;
    struct TreeNode n = nodes[idx];
    return n.value + tree_sum(nodes, n.left) + tree_sum(nodes, n.right);
}

__global__ void tree_sum_kernel(int *out, struct TreeNode *nodes, int root, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = tree_sum(nodes, root);
}

// ------------------------------------------------------------------
// Maximum complexity kernel: uses shared memory, warp shuffles,
// atomics, device function calls, structs, ternary, all in one.

struct Stats2 { float mean; float var; };

__device__ struct Stats2 compute_stats(float *data, int len) {
    struct Stats2 s;
    float sum = 0.0f, sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += data[i];
        sq += data[i] * data[i];
    }
    s.mean = sum / (float)len;
    s.var = sq / (float)len - s.mean * s.mean;
    return s;
}

__global__ void mega_kernel(float *out, float *in, int *flags, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;

    // Step 1: Load with conditional
    float v = (gid < n) ? in[gid] : 0.0f;
    v = (flags[gid % n] > 0) ? v * 2.0f : -v;

    // Step 2: Warp reduce
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFF, v,  1);

    // Step 3: Store warp sums to shared
    if (lane == 0) smem[tid >> 5] = v;
    __syncthreads();

    // Step 4: First warp reduces warp sums
    if (tid < 8) {
        float ws = smem[tid];
        ws += __shfl_xor_sync(0xFF, ws, 4);
        ws += __shfl_xor_sync(0xFF, ws, 2);
        ws += __shfl_xor_sync(0xFF, ws, 1);
        if (tid == 0) smem[0] = ws;
    }
    __syncthreads();

    // Step 5: Broadcast block sum and compute normalized output
    float block_sum = smem[0];
    float orig = (gid < n) ? in[gid] : 0.0f;
    float normalized = (block_sum != 0.0f) ? orig / block_sum : 0.0f;

    // Step 6: Apply activation
    float result = (normalized > 0.0f) ? normalized : 0.01f * normalized;  // leaky relu

    if (gid < n) out[gid] = result;
}

// ------------------------------------------------------------------
// Kernel mixing recursive + non-recursive device function calls.

__device__ int recursive_depth(int n) {
    if (n <= 0) return 0;
    return 1 + recursive_depth(n / 2);
}

__device__ float nonrec_scale(float x, float factor) {
    return x * factor + 1.0f;
}

__global__ void mixed_calls(float *out, int *in_i, float *in_f, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int depth = recursive_depth(in_i[tid]);
        float scaled = nonrec_scale(in_f[tid], (float)depth);
        out[tid] = scaled;
    }
}

// ------------------------------------------------------------------
// All integer sizes in one kernel: char, short, int, long long.

__global__ void all_int_sizes(long long *out,
                                 signed char *c_in, short *s_in,
                                 int *i_in, long long *ll_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long r = (long long)c_in[tid]
                    + (long long)s_in[tid]
                    + (long long)i_in[tid]
                    + ll_in[tid];
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// All float sizes in one kernel: half, float, double.

__global__ void all_float_sizes(double *out,
                                   unsigned short *h_in, float *f_in,
                                   double *d_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half h = __ushort_as_half(h_in[tid]);
        float hf = __half2float(h);
        double r = (double)hf + (double)f_in[tid] + d_in[tid];
        out[tid] = r;
    }
}
