// Probe: real CUDA algorithm implementations that stress the compiler
// end-to-end: Bellman-Ford, BFS frontier, radix sort step, SpMV,
// and parallel prefix with arbitrary type.

// ------------------------------------------------------------------
// Bellman-Ford iteration step: relax edges.

struct Edge { int src, dst; float weight; };

__global__ void bellman_ford_relax(float *dist, struct Edge *edges,
                                    int num_edges, int *updated) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid < num_edges) {
        int s = edges[eid].src;
        int d = edges[eid].dst;
        float w = edges[eid].weight;
        float new_d = dist[s] + w;
        if (new_d < dist[d]) {
            dist[d] = new_d;   // race condition is OK for convergence test
            *updated = 1;
        }
    }
}

// ------------------------------------------------------------------
// BFS frontier expansion: for each frontier node, enqueue neighbors.

__global__ void bfs_expand(int *frontier_in, int frontier_size,
                            int *frontier_out, int *out_size,
                            int *adj, int *adj_offsets,
                            int *visited, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int node = frontier_in[tid];
        int start = adj_offsets[node];
        int end   = adj_offsets[node + 1];
        for (int e = start; e < end; e++) {
            int nbr = adj[e];
            if (visited[nbr] == 0) {
                visited[nbr] = level;
                int pos = atomicAdd(out_size, 1);
                frontier_out[pos] = nbr;
            }
        }
    }
}

// ------------------------------------------------------------------
// Radix sort step: compute histogram for one digit.

__global__ void radix_histogram(int *hist, int *in, int n, int bit) {
    __shared__ int local_hist[16];  // 4-bit radix → 16 buckets
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Clear shared histogram
    if (tid < 16) local_hist[tid] = 0;
    __syncthreads();

    if (gid < n) {
        int digit = (in[gid] >> bit) & 0xF;
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write to global histogram
    if (tid < 16) {
        atomicAdd(&hist[blockIdx.x * 16 + tid], local_hist[tid]);
    }
}

// ------------------------------------------------------------------
// SpMV: sparse matrix-vector multiplication (CSR format).

__global__ void spmv_csr(float *y, float *values, int *col_idx, int *row_ptr,
                          float *x, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];
        for (int j = row_start; j < row_end; j++) {
            dot += values[j] * x[col_idx[j]];
        }
        y[row] = dot;
    }
}

// ------------------------------------------------------------------
// K-means assignment step: assign each point to nearest centroid.

__global__ void kmeans_assign(int *assignments, float *points, float *centroids,
                               int n_points, int n_centroids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_points) {
        float px = points[tid * 2 + 0];
        float py = points[tid * 2 + 1];
        float best_dist = 3.402823466e+38f;
        int best_k = 0;
        for (int k = 0; k < n_centroids; k++) {
            float cx = centroids[k * 2 + 0];
            float cy = centroids[k * 2 + 1];
            float dx = px - cx;
            float dy = py - cy;
            float d = dx*dx + dy*dy;
            if (d < best_dist) {
                best_dist = d;
                best_k = k;
            }
        }
        assignments[tid] = best_k;
    }
}

// ------------------------------------------------------------------
// Merge sort step: merge two sorted halves.

__global__ void merge_step(int *out, int *in, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block = tid / stride;
    int pos   = tid % stride;
    int left  = block * (2 * stride);
    int mid   = left + stride;
    int right = left + 2 * stride;
    if (right > n) right = n;
    if (mid > n)   mid = n;
    if (left >= n) return;

    // Binary search for position in merged output
    int lo = 0, hi = mid - left;
    int val = (pos < right - left) ? in[left + pos] : 2147483647;
    // Find insertion point in the other half
    int other_half_size = right - mid;
    int lb = 0, ub = other_half_size;
    while (lb < ub) {
        int m = (lb + ub) / 2;
        if (in[mid + m] < val) lb = m + 1;
        else ub = m;
    }
    if (pos < right - left) {
        out[left + pos + lb] = val;
    }
}

// ------------------------------------------------------------------
// Jaccard similarity: |A ∩ B| / |A ∪ B| for bit sets.

__device__ float jaccard(unsigned int *A, unsigned int *B, int words) {
    int intersect = 0, union_ab = 0;
    for (int w = 0; w < words; w++) {
        intersect += __popc(A[w] & B[w]);
        union_ab  += __popc(A[w] | B[w]);
    }
    return (union_ab > 0) ? (float)intersect / (float)union_ab : 1.0f;
}

__global__ void jaccard_kernel(float *out, unsigned int *sets, int n_sets, int words) {
    int tid = threadIdx.x;
    if (tid < n_sets) {
        // Compare each set against set 0
        out[tid] = jaccard(sets + tid * words, sets, words);
    }
}
