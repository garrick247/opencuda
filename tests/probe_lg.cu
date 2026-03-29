// Probe: __ldg (read-only cache), extern __shared__ (dynamic shared mem),
// conditional with float comparison, pointer-typed loop index,
// multiple assignments to same variable in sequential ifs

// __ldg read-only cache loads
__global__ void ldg_sum(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = __ldg(&in[tid]);
        out[tid] = val * 2;
    }
}

// extern __shared__ (dynamic shared memory — size passed at launch)
__global__ void dynamic_shared(int *out, int n) {
    extern __shared__ int smem[];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = tid * tid;
        __syncthreads();
        out[tid] = smem[tid];
    }
}

// Conditional reassignment: same variable assigned in both branches
__global__ void conditional_assign(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float result;
        if (v > 0.0f) {
            result = v * 2.0f;
        } else {
            result = -v;
        }
        out[tid] = result;
    }
}

// Chained conditionals: value set by first matching branch
__global__ void classify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int category = 0;
        if (x < 0) category = -1;
        if (x == 0) category = 0;
        if (x > 0) category = 1;
        out[tid] = category;
    }
}

// Multiple return values via output pointers
__global__ void minmax_kernel(int *out_min, int *out_max, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mn = in[0];
        int mx = in[0];
        for (int i = 1; i < n; i++) {
            int v = in[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        *out_min = mn;
        *out_max = mx;
    }
}
