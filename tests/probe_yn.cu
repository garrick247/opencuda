// Probe: stress tests for address-taken locals in complex control flow,
// multi-pointer local (two &vars), variable used as both value and address,
// loop where &var is taken and used across iterations, struct returned
// and immediately field-accessed, chain of struct returns, and kernel
// that has both __shared__ and __constant__ memory access patterns.

// ------------------------------------------------------------------
// Address-taken local in loop (classic ring-buffer or sliding window).

__global__ void addr_taken_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int window[4] = {0, 0, 0, 0};
        int *slot = window;  // pointer into local array
        int total = 0;
        for (int i = 0; i < n; i++) {
            *slot = in[i];
            slot++;
            if (slot >= window + 4) slot = window;  // wrap
            // Sum current window
            int s = 0;
            for (int k = 0; k < 4; k++) s += window[k];
            total = s;
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Two address-taken locals passed to same function.

__device__ void sort2(int *a, int *b) {
    if (*a > *b) {
        int tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

__global__ void sort2_kernel(int *out_lo, int *out_hi, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid];
        sort2(&x, &y);
        out_lo[tid] = x;
        out_hi[tid] = y;
    }
}

// ------------------------------------------------------------------
// Variable used as both value (read) and address (write via pointer).

__global__ void val_and_addr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int *p = &v;
        int r = v * 2;   // read v as value
        *p = r + 1;       // write v through pointer
        out[tid] = v;     // read v again (should be r+1)
    }
}

// ------------------------------------------------------------------
// Struct returned from __device__ and immediately field-accessed.

struct MinMax { int lo, hi; };

__device__ struct MinMax minmax_arr(int *arr, int len) {
    struct MinMax mm;
    mm.lo = arr[0];
    mm.hi = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] < mm.lo) mm.lo = arr[i];
        if (arr[i] > mm.hi) mm.hi = arr[i];
    }
    return mm;
}

__global__ void minmax_immediate(int *out_range, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Build local window
        int w[4];
        for (int k = 0; k < 4; k++) w[k] = in[(tid + k) % n];
        // Use returned struct's fields immediately
        int range = minmax_arr(w, 4).hi - minmax_arr(w, 4).lo;
        out_range[tid] = range;
    }
}

// ------------------------------------------------------------------
// Chain of struct returns: f returns struct used as arg to g.

struct Weighted { float val; int weight; };

__device__ struct Weighted make_weighted(float v, int w) {
    struct Weighted r;
    r.val    = v;
    r.weight = w;
    return r;
}

__device__ float weighted_sum(struct Weighted a, struct Weighted b) {
    return a.val * (float)a.weight + b.val * (float)b.weight;
}

__global__ void chain_struct(float *out, float *v, int *w, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Weighted a = make_weighted(v[tid],       w[tid]);
        struct Weighted b = make_weighted(v[tid] * 2.0f, w[tid] + 1);
        out[tid] = weighted_sum(a, b);
    }
}

// ------------------------------------------------------------------
// Kernel using both __shared__ and __constant__.

__constant__ float c_kernel[3] = {0.25f, 0.5f, 0.25f};

__global__ void shared_and_const(float *out, float *in, int n) {
    __shared__ float smem[258];  // 256 + 1 halo on each side
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // Load with halo
    smem[tid + 1] = (gid < n) ? in[gid] : 0.0f;
    if (tid == 0)                       smem[0]   = (gid > 0) ? in[gid-1] : 0.0f;
    if (tid == blockDim.x - 1)          smem[blockDim.x+1] = (gid+1 < n) ? in[gid+1] : 0.0f;
    __syncthreads();
    // Convolve with __constant__ 3-tap kernel
    if (gid < n) {
        out[gid] = smem[tid] * c_kernel[0]
                 + smem[tid+1] * c_kernel[1]
                 + smem[tid+2] * c_kernel[2];
    }
}

// ------------------------------------------------------------------
// Local array passed to __device__ function (pointer param).

__device__ int array_sum(int *arr, int len) {
    int s = 0;
    for (int i = 0; i < len; i++) s += arr[i];
    return s;
}

__global__ void local_arr_to_func(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[8];
        for (int k = 0; k < 8; k++) buf[k] = in[(tid + k) % n];
        out[tid] = array_sum(buf, 8);
    }
}
