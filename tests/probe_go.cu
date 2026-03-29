// Probe: real-world BFS/graph algorithm patterns
// Frontier-based BFS, adjacency list with CSR format

__global__ void bfs_frontier(int *out_dist, int *frontier, int *next_frontier,
                               int *frontier_size, int *next_size,
                               int *row_ptr, int *col_idx, int dist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *frontier_size) return;

    int u = frontier[tid];
    int start = row_ptr[u];
    int end = row_ptr[u + 1];

    for (int e = start; e < end; e++) {
        int v = col_idx[e];
        if (out_dist[v] == -1) {
            out_dist[v] = dist;
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// SSSP (Single Source Shortest Path) Bellman-Ford step
__global__ void bellman_ford_step(float *dist, int *src_arr, int *dst_arr,
                                    float *weight, int m, int *updated) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= m) return;

    int u = src_arr[eid];
    int v = dst_arr[eid];
    float w = weight[eid];

    if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        *updated = 1;
    }
}

// Connected components — union-find with path compression
__device__ int find_root(int *parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // path compression
        x = parent[x];
    }
    return x;
}

__global__ void union_find_step(int *parent, int *edges_u, int *edges_v, int m) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= m) return;

    int ru = find_root(parent, edges_u[eid]);
    int rv = find_root(parent, edges_v[eid]);
    if (ru != rv) {
        if (ru < rv) {
            atomicMin(&parent[rv], ru);
        } else {
            atomicMin(&parent[ru], rv);
        }
    }
}
