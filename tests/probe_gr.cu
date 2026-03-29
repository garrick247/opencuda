// Probe: Edge case in unary operators — ++/-- on pointer, *p++, (*p)++,
// multiple pre/post increments in same expression (UB in C but should parse)

__global__ void ptr_incr(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = in + tid;
        // Post-increment pointer
        float v1 = *p;
        p++;
        float v2 = (p < in + n) ? *p : 0.0f;
        out[tid] = v1 + v2;
    }
}

// Pre-increment in subscript
__global__ void pre_incr_subscript(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int sum = 0;
        while (i < n - 1) {
            sum += in[++i];  // pre-increment used as index
        }
        out[tid] = sum;
    }
}

// Compound: (*p)++ — increment value at pointer
__global__ void deref_incr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Make a local copy and increment
        int val = in[tid];
        val++;
        out[tid] = val;
    }
}

// Bitwise NOT (~) on various types
__global__ void bitwise_not(int *out_i, unsigned int *out_u,
                              int *in_i, unsigned int *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_i[tid] = ~in_i[tid];
        out_u[tid] = ~in_u[tid];
    }
}
