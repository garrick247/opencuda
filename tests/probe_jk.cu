// Probe: variable shadowing (inner scope re-declares name used by outer),
// conditional stores, __syncthreads with __shared__ accumulation,
// outer variable survives inner-scope redeclaration

// Variable shadowing: outer 'x' must retain value after loop with inner 'x'
__global__ void shadow_outer(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = 42;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int x = in[i];   // inner x shadows outer x
            sum += x;        // uses inner x
        }
        out[0] = sum;
        out[1] = x;          // should be 42, not last in[i]
    }
}

// Conditional store: only write when condition holds
__global__ void conditional_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > 0) {
            out[tid] = v * 2;
        }
        // else: out[tid] unchanged (not written)
    }
}

// __syncthreads with __shared__: each thread writes its value, sync, then
// thread 0 sums all
__global__ void shared_reduce(int *out, int *in, int n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;
    if (tid < n) {
        buf[tid] = in[tid];
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += buf[i];
        }
        *out = sum;
    }
}

// Nested scope with same variable name in two sibling if-blocks
__global__ void sibling_scope(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        if (v > 0) {
            int tmp = v * 2;
            result = tmp + 1;
        }
        if (v < 0) {
            int tmp = v * (-1);   // same name 'tmp' in sibling if
            result = tmp - 1;
        }
        out[tid] = result;
    }
}
