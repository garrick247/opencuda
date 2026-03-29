// Probe: const variable folding, #define macro in loop bounds,
// unsigned overflow wrap-around, signed/unsigned comparison edge cases,
// const pointer arithmetic

#define WARP_SIZE 32
#define MAX_ITER 8

// const variable should be folded to compile-time constant
__global__ void const_fold_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const float scale = 2.5f;     // should be folded
        const int   shift = 3;        // should be folded
        float v = in[tid];
        v = v * scale + (float)shift;
        out[tid] = v;
    }
}

// #define constant in loop bound
__global__ void macro_loop(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < MAX_ITER; i++) {
            sum += in[i];
        }
        *out = sum;
    }
}

// Unsigned wrap-around: 0u - 1 should wrap to UINT_MAX
__global__ void unsigned_wrap(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Wrapping subtraction: wraps at 0
        unsigned int w = v - 1u;
        out[tid] = w;
    }
}

// WARP_SIZE #define in lane computation
__global__ void warp_lane(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    if (tid < n) {
        out[tid] = in[tid] + lane;
    }
}

// Pointer to const data (const int *) — data pointed to is const, not pointer
__global__ void const_ptr_read(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2;
    }
}
