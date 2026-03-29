// Probe: multiple assignment targets, chained assignments
// - a = b = c = 0;
// - conditional assignment inside loop: if (x) a = 1; else a = 2;
// - function return used as array index
// - Casting return value of device function

__device__ int clamp_idx(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void chained_assign(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = b = c = tid;
        out[a] = b + c;
    }
}

__global__ void func_as_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = clamp_idx(tid * 3, 0, n - 1);
        out[tid] = in[idx];
    }
}

__global__ void nested_ternary(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid];
        int result = (x > 0) ? ((y > 0) ? x + y : x - y) : ((y > 0) ? y - x : -(x + y));
        out[tid] = result;
    }
}
