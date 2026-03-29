// Probe: Correctness of loop-carried values across multiple loop iterations
// - Loop with multiple carry variables that interact
// - Loop where modification of one var affects condition involving another
// - Loop-carried pointer (advancing through array)
// - Reduction with different operators (max, min, sum, product)

__global__ void multi_reduce(float *out_sum, float *out_max, float *out_min,
                              float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {  // Single thread does sequential reduction
        float sum = 0.0f;
        float mx = in[0];
        float mn = in[0];
        for (int i = 0; i < n; i++) {
            float v = in[i];
            sum += v;
            if (v > mx) mx = v;
            if (v < mn) mn = v;
        }
        *out_sum = sum;
        *out_max = mx;
        *out_min = mn;
    }
}

// Loop with pointer advancement
__global__ void ptr_advance_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        float *p = in;
        float *end = in + n;
        float sum = 0.0f;
        while (p < end) {
            sum += *p;
            p++;
        }
        *out = sum;
    }
}

// Running gcd (Euclidean algorithm)
__device__ int gcd(int a, int b) {
    while (b != 0) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

__global__ void gcd_array(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = gcd(a[tid], b[tid]);
    }
}
