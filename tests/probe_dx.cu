// Probe: Complex __device__ inlining edge cases
// - Device function with early return, called multiple times in same kernel
// - Device function that has a loop inside, called inside a loop in kernel
// - Two different device functions called with same argument
// - Device function returning void used as statement

__device__ float abs_clamp(float x, float limit) {
    float a = (x < 0.0f) ? -x : x;
    return (a > limit) ? limit : a;
}

__device__ void write_pair(float *out, int idx, float lo, float hi) {
    out[idx * 2]     = lo;
    out[idx * 2 + 1] = hi;
}

__global__ void inline_stress(float *out, float *in, float limit, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Call same device function twice with different args
        float a1 = abs_clamp(v, limit);
        float a2 = abs_clamp(v * 2.0f, limit);
        // Call another device function (void, no return used)
        write_pair(out, tid, a1, a2);
        // Call abs_clamp inside conditional
        float check = abs_clamp(a1 - a2, 1.0f);
        out[tid * 2] += check;
    }
}

// Device function called inside loop
__device__ int is_prime_simple(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

__global__ void count_primes(int *out, int start, int count) {
    int tid = threadIdx.x;
    if (tid < 1) {
        int total = 0;
        for (int i = start; i < start + count; i++) {
            if (is_prime_simple(i)) total++;
        }
        out[0] = total;
    }
}
