// Probe: advanced patterns — warp-level matrix accumulate (manual),
// cooperative reduction across warps via shared atomics, parallel
// histogram with privatization, n-body gravitational force,
// prefix sum with carry propagation, and block-level transpose.

// ------------------------------------------------------------------
// N-body gravitational force computation.

__global__ void nbody_force(float *fx, float *fy, float *fz,
                               float *px, float *py, float *pz,
                               float *mass, float eps2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    float xi = px[i], yi = py[i], zi = pz[i];
    for (int j = 0; j < n; j++) {
        float dx = px[j] - xi;
        float dy = py[j] - yi;
        float dz = pz[j] - zi;
        float dist2 = dx*dx + dy*dy + dz*dz + eps2;
        float inv_dist = rsqrtf(dist2);
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float mj = mass[j];
        ax += mj * dx * inv_dist3;
        ay += mj * dy * inv_dist3;
        az += mj * dz * inv_dist3;
    }
    fx[i] = ax; fy[i] = ay; fz[i] = az;
}

// ------------------------------------------------------------------
// Privatized histogram (per-thread private bins, then reduce).

__global__ void priv_histogram(int *global_hist, int *data, int bins, int n) {
    __shared__ int shared_hist[256];
    int tid = threadIdx.x;
    // Zero shared histogram
    if (tid < bins) shared_hist[tid] = 0;
    __syncthreads();
    // Private accumulation
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        int b = data[i] % bins;
        if (b >= 0 && b < bins) atomicAdd(&shared_hist[b], 1);
    }
    __syncthreads();
    // Merge to global
    if (tid < bins) atomicAdd(&global_hist[tid], shared_hist[tid]);
}

// ------------------------------------------------------------------
// Cooperative block reduction: max with index (argmax).

__global__ void argmax_block(float *out_val, int *out_idx,
                                float *in, int n) {
    __shared__ float sval[256];
    __shared__ int   sidx[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sval[tid] = (gid < n) ? in[gid] : -1e30f;
    sidx[tid] = (gid < n) ? gid : -1;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sval[tid + s] > sval[tid]) {
            sval[tid] = sval[tid + s];
            sidx[tid] = sidx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_val[blockIdx.x] = sval[0];
        out_idx[blockIdx.x] = sidx[0];
    }
}

// ------------------------------------------------------------------
// Image processing: Sobel edge detection (3x3 gradient magnitude).

__global__ void sobel(float *out, float *in, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) {
        if (x < W && y < H) out[y * W + x] = 0.0f;
        return;
    }
    // Sobel kernels
    float gx = -in[(y-1)*W+(x-1)] + in[(y-1)*W+(x+1)]
              -2*in[y*W+(x-1)]     + 2*in[y*W+(x+1)]
              -in[(y+1)*W+(x-1)]   + in[(y+1)*W+(x+1)];
    float gy = -in[(y-1)*W+(x-1)] - 2*in[(y-1)*W+x] - in[(y-1)*W+(x+1)]
              +in[(y+1)*W+(x-1)]   + 2*in[(y+1)*W+x] + in[(y+1)*W+(x+1)];
    out[y * W + x] = sqrtf(gx*gx + gy*gy);
}

// ------------------------------------------------------------------
// Block-level transpose using shared memory.

__global__ void block_transpose(float *out, float *in, int W, int H) {
    __shared__ float tile[16][17];  // +1 padding to avoid bank conflicts
    int bx = blockIdx.x * 16, by = blockIdx.y * 16;
    int tx = threadIdx.x, ty = threadIdx.y;
    // Read coalesced
    int ix = bx + tx, iy = by + ty;
    if (ix < W && iy < H) tile[ty][tx] = in[iy * W + ix];
    __syncthreads();
    // Write transposed
    int ox = by + tx, oy = bx + ty;
    if (ox < H && oy < W) out[oy * H + ox] = tile[tx][ty];
}

// ------------------------------------------------------------------
// Mandelbrot set computation.

__global__ void mandelbrot(int *out, float x0, float y0,
                              float dx, float dy, int W, int H, int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;
    float cr = x0 + px * dx;
    float ci = y0 + py * dy;
    float zr = 0.0f, zi = 0.0f;
    int iter = 0;
    while (zr*zr + zi*zi < 4.0f && iter < max_iter) {
        float tmp = zr*zr - zi*zi + cr;
        zi = 2.0f*zr*zi + ci;
        zr = tmp;
        iter++;
    }
    out[py * W + px] = iter;
}

// ------------------------------------------------------------------
// Parallel prefix product (instead of sum).

__global__ void prefix_product(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 1.0f;
    __syncthreads();
    for (int d = 1; d < blockDim.x; d <<= 1) {
        float val = (tid >= d) ? smem[tid - d] : 1.0f;
        __syncthreads();
        smem[tid] *= val;
        __syncthreads();
    }
    if (gid < n) out[gid] = smem[tid];
}
