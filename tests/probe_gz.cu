// Probe: Final stress test — everything at once
// Large kernel with: struct, enum, template macros, 64-bit, atomics,
// shared memory, warp shuffles, complex control flow

#define MAX_LEVELS 8
#define WARP_SZ 32

enum NodeType { LEAF = 0, INTERNAL = 1, ROOT = 2 };

struct TreeNode {
    int parent;
    int left, right;
    float value;
    int level;
    int type;
};

__device__ float reduce_warp(float val) {
    unsigned mask = 0xFFFFFFFF;
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ int find_depth(TreeNode *nodes, int idx) {
    int depth = 0;
    while (nodes[idx].parent != -1 && depth < MAX_LEVELS) {
        idx = nodes[idx].parent;
        depth++;
    }
    return depth;
}

__global__ void tree_stats(float *out_sum, float *out_mean, int *out_depth,
                             TreeNode *nodes, int n) {
    __shared__ float warp_sums[8];  // one per warp

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int warp = tid / WARP_SZ;
    int lane = tid % WARP_SZ;

    float local_val = (gid < n) ? nodes[gid].value : 0.0f;
    float warp_sum = reduce_warp(local_val);

    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        float block_sum = reduce_warp(warp_sums[lane]);
        if (lane == 0) {
            atomicAdd(out_sum, block_sum);
        }
    }

    if (gid < n) {
        int depth = find_depth(nodes, gid);
        out_depth[gid] = depth;
        int type = nodes[gid].type;
        if (type == ROOT) {
            out_mean[0] = nodes[gid].value;
        }
    }
}
