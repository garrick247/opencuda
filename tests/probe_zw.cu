// Probe: graph algorithms + computational geometry — BFS level step,
// connected components (label propagation), convex hull orientation test,
// point-in-polygon, k-d tree distance, and sparse attention mask.

// ------------------------------------------------------------------
// BFS level step: advance frontier by one level.

__global__ void bfs_step(int *level, int *adj_list, int *adj_offset,
                            int *frontier, int *next_frontier,
                            int *next_count, int current_level, int n_frontier) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_frontier) return;
    int node = frontier[gid];
    int start = adj_offset[node];
    int end   = adj_offset[node + 1];
    for (int e = start; e < end; e++) {
        int neighbor = adj_list[e];
        // atomicCAS to claim unvisited neighbor
        int old = atomicCAS(&level[neighbor], -1, current_level + 1);
        if (old == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

// ------------------------------------------------------------------
// Label propagation (connected components).

__global__ void label_propagate(int *labels, int *adj_list, int *adj_offset,
                                   int n_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    int my_label = labels[node];
    int start = adj_offset[node];
    int end   = adj_offset[node + 1];
    int min_label = my_label;
    for (int e = start; e < end; e++) {
        int neighbor_label = labels[adj_list[e]];
        if (neighbor_label < min_label) min_label = neighbor_label;
    }
    if (min_label < my_label) labels[node] = min_label;
}

// ------------------------------------------------------------------
// Orientation test: cross product sign for convex hull.

__device__ int orientation(float px, float py, float qx, float qy,
                             float rx, float ry) {
    float val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy);
    if (val > 0.0001f) return 1;   // counterclockwise
    if (val < -0.0001f) return -1; // clockwise
    return 0;                        // collinear
}

__global__ void orient_test(int *out, float *px, float *py,
                               float *qx, float *qy,
                               float *rx, float *ry, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = orientation(px[tid], py[tid], qx[tid], qy[tid],
                               rx[tid], ry[tid]);
    }
}

// ------------------------------------------------------------------
// Point-in-polygon (ray casting).

__global__ void point_in_polygon(int *out, float *test_x, float *test_y,
                                    float *poly_x, float *poly_y,
                                    int n_verts, int n_points) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_points) return;
    float tx = test_x[gid], ty = test_y[gid];
    int inside = 0;
    for (int i = 0, j = n_verts - 1; i < n_verts; j = i++) {
        float xi = poly_x[i], yi = poly_y[i];
        float xj = poly_x[j], yj = poly_y[j];
        if ((yi > ty) != (yj > ty)) {
            float x_intersect = (xj - xi) * (ty - yi) / (yj - yi) + xi;
            if (tx < x_intersect) inside = !inside;
        }
    }
    out[gid] = inside;
}

// ------------------------------------------------------------------
// Sparse attention: only compute scores for non-zero mask positions.

__global__ void sparse_attention(float *out, float *Q, float *K,
                                    int *mask_row, int *mask_col,
                                    int n_nonzero, int head_dim) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_nonzero) return;
    int q_pos = mask_row[gid];
    int k_pos = mask_col[gid];
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += Q[q_pos * head_dim + d] * K[k_pos * head_dim + d];
    }
    out[gid] = dot * rsqrtf((float)head_dim);
}

// ------------------------------------------------------------------
// Distance matrix computation.

__global__ void distance_matrix(float *D, float *X, int dim, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    float s = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = X[i * dim + d] - X[j * dim + d];
        s += diff * diff;
    }
    D[i * n + j] = sqrtf(s);
}
