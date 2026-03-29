// Probe: warp shuffle variants (shfl_xor/up/down), cache hint loads
// (__ldcg/__ldcs/__ldlu), __syncwarp, __threadfence_system, and
// warp-level reduction patterns.

// ------------------------------------------------------------------
// __shfl_xor_sync: butterfly exchange for warp reduction.

__global__ void shfl_xor_reduce(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (gid < n) ? in[gid] : 0.0f;

    // Butterfly warp reduction
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  8);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  4);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  2);
    val += __shfl_xor_sync(0xFFFFFFFF, val,  1);

    if (threadIdx.x % 32 == 0 && gid < n) {
        out[gid / 32] = val;
    }
}

// ------------------------------------------------------------------
// __shfl_up_sync: prefix scan step.

__global__ void shfl_up_scan(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int val = (gid < n) ? in[gid] : 0;

    // Warp-level inclusive scan using shfl_up
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane >= offset) val += tmp;
    }

    if (gid < n) out[gid] = val;
}

// ------------------------------------------------------------------
// __shfl_down_sync: broadcast from lane 0.

__global__ void shfl_down_bcast(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int val = (gid < n) ? in[gid] : 0;

    // Broadcast lane 0's value to all lanes
    int lane0_val = __shfl_down_sync(0xFFFFFFFF, val, lane);

    if (gid < n) out[gid] = lane0_val;
}

// ------------------------------------------------------------------
// __shfl_sync (direct lane read).

__global__ void shfl_lane(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int val = in[gid];
        // Each thread reads lane 0's value
        int lane0 = __shfl_sync(0xFFFFFFFF, val, 0);
        out[gid] = lane0;
    }
}

// ------------------------------------------------------------------
// __ldcg: cache-global load hint.

__global__ void ldcg_load(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        float v = __ldcg(in + gid);
        out[gid] = v * 2.0f;
    }
}

// ------------------------------------------------------------------
// __ldcs: cache-streaming load hint.

__global__ void ldcs_load(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float v = __ldcs(in + gid);
        out[gid] = v + 1.0f;
    }
}

// ------------------------------------------------------------------
// __syncwarp(): synchronize threads within a warp.

__global__ void syncwarp_kernel(int *out, int *in, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        smem[tid & 31] = in[gid];
    }
    __syncwarp();

    if (gid < n) {
        int neighbor = smem[(tid + 1) & 31];
        out[gid] = in[gid] + neighbor;
    }
}

// ------------------------------------------------------------------
// __threadfence_system(): system-wide memory fence.

__global__ void fence_system(int *out, volatile int *flag, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = tid;
        __threadfence_system();
        *flag = 1;  // signal completion after fence
    }
}

// ------------------------------------------------------------------
// Warp lane ID via threadIdx.x & 31.

__global__ void warp_lane_id(int *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        out[gid] = lane + warp * 32;   // == threadIdx.x
    }
}
