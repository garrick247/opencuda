// Probe: warp-cooperative algorithms — segmented scan, warp-strided loop,
// ballot-based compaction, conditional global write, warp-level divergent
// branch behavior, grid-stride loop pattern, and cooperative block reduction
// with multiple levels of shared memory.

// ------------------------------------------------------------------
// Grid-stride loop (single kernel handles arbitrarily large arrays).

__global__ void grid_stride(float *out, float *in, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = in[i] * in[i];
    }
}

// ------------------------------------------------------------------
// Ballot-based compaction: write only threads with v > 0 to output.

__global__ void ballot_compact(int *out, int *out_count, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int v = (gid < n) ? in[gid] : 0;
    // Ballot: which lanes have v > 0
    unsigned mask = __ballot_sync(0xFFFFFFFF, v > 0);
    // Prefix popcount for this lane's rank
    unsigned below = mask & ((1u << lane) - 1u);
    int rank = __popc(below);
    // Each warp writes its active count once
    int warp_base = 0;
    if (lane == 0) warp_base = atomicAdd(out_count, __popc(mask));
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);
    // Write compacted
    if ((v > 0) && gid < n) out[warp_base + rank] = v;
}

// ------------------------------------------------------------------
// Segmented scan: reset accumulator at segment boundary.

__global__ void segmented_scan(int *out, int *in, int *seg_id, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    // Simple serial segmented prefix sum (each thread does its segment)
    // This tests that the per-thread view works:
    int v = in[gid];
    int seg = seg_id[gid];
    int s = v;
    // Look at previous elements in same segment (capped at 4)
    for (int d = 1; d <= 4 && gid - d >= 0; d++) {
        if (seg_id[gid - d] == seg) s += in[gid - d];
        else break;
    }
    out[gid] = s;
}

// ------------------------------------------------------------------
// Warp-strided loop: each warp iterates with stride 32.

__global__ void warp_strided(float *out, float *in, int n) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane    = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    float s = 0.0f;
    for (int i = warp_id; i < n; i += total_warps) {
        // Each warp reads n/total_warps elements, one per lane
        int idx = i * 32 + lane;
        if (idx < n) s += in[idx];
    }
    // Reduce across warp
    s += __shfl_xor_sync(0xFFFFFFFF, s, 16);
    s += __shfl_xor_sync(0xFFFFFFFF, s,  8);
    s += __shfl_xor_sync(0xFFFFFFFF, s,  4);
    s += __shfl_xor_sync(0xFFFFFFFF, s,  2);
    s += __shfl_xor_sync(0xFFFFFFFF, s,  1);
    if (lane == 0) out[warp_id] = s;
}

// ------------------------------------------------------------------
// Conditional global write (predicated store).

__global__ void pred_store(int *out, int *in, int threshold, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > threshold) out[tid] = v;
        // else: don't write (previous value preserved)
    }
}

// ------------------------------------------------------------------
// Block reduction using shared memory: max reduction.

__global__ void block_max(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0x80000000;  // INT_MIN
    __syncthreads();
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid])
            smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Multi-output per-thread: write to 3 output arrays in one pass.

__global__ void multi_output(float *out1, float *out2, float *out3,
                               float *in1, float *in2, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in1[tid], b = in2[tid];
        out1[tid] = a + b;
        out2[tid] = a * b;
        out3[tid] = a - b;
    }
}

// ------------------------------------------------------------------
// Fused multiply-add chain (DFMA style in integer).

__global__ void int_fma_chain(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        int z = c[tid];
        // a*b + c, a*c + b, b*c + a
        int r1 = x * y + z;
        int r2 = x * z + y;
        int r3 = y * z + x;
        out[tid] = r1 ^ r2 ^ r3;  // XOR to avoid dead-code elimination
    }
}
