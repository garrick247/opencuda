// Probe: function pointer typedef, casting through void*, 
//        complex return type chains, multiple levels of struct nesting

// Chained function calls as expressions
__device__ float fminf3(float a, float b, float c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

__device__ float fmaxf3(float a, float b, float c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

// Pre-increment in expression context (not as statement)
__global__ void preinc_expr(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = tid;
        int v = ++i;      // pre-increment: v = i+1
        out[tid] = v + i; // should be (tid+1) + (tid+1) = 2*tid+2
    }
}

// Post-increment in expression context
__global__ void postinc_expr(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = tid;
        int v = i++;      // post-increment: v = tid, i = tid+1
        out[tid] = v + i; // should be tid + (tid+1) = 2*tid+1
    }
}

// Chained function calls
__global__ void chained_calls(float *out, float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = fminf3(a[tid], b[tid], c[tid]) + fmaxf3(a[tid], b[tid], c[tid]);
    }
}
