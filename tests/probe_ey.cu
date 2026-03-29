// Probe: multiple #define macros, macro with parameters, macro used in
// array size, loop bound, and expressions

#define MAX_THREADS 256
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define CLAMP(v, lo, hi) MAX(lo, MIN(hi, v))
#define SQ(x) ((x) * (x))
#define BLOCK_DIM 32
#define ELEMENTS_PER_THREAD 4

__global__ void macro_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    int base = tid * ELEMENTS_PER_THREAD;

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = MIN(base + i, n - 1);
        float v = in[idx];
        float clamped = CLAMP(v, -1.0f, 1.0f);
        out[base + i] = SQ(clamped);
    }
}

#define WARP_SIZE 32
#define LOG2_WARP 5

__global__ void warp_scan(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    if (tid < n) {
        int val = in[tid];
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            int t = __shfl_up_sync(0xFFFFFFFF, val, d);
            if (lane >= d) val += t;
        }
        out[tid] = val;
    }
}
