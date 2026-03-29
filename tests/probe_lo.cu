// Probe: multiple accumulators in same loop,
// reduction tree (parallel prefix-style),
// loop with conditional accumulation (if inside loop),
// mixed types in loop (int and float accumulators)

// Three independent accumulators in same loop
__global__ void triple_accumulate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0, product = 1, count = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            sum += v;
            if (v != 0) product *= v;  // only multiply non-zero
            if (v > 0) count++;         // count positives
        }
        out[0] = sum;
        out[1] = product;
        out[2] = count;
    }
}

// Parallel scan step: each thread computes partial result
__global__ void prefix_step(int *out, int *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n && tid >= stride) {
        out[tid] = in[tid] + in[tid - stride];
    } else if (tid < n) {
        out[tid] = in[tid];
    }
}

// Loop with conditional: sum only positive elements
__global__ void cond_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_sum = 0, neg_sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) {
                pos_sum += v;
            } else {
                neg_sum += v;
            }
        }
        out[0] = pos_sum;
        out[1] = neg_sum;
    }
}

// Mixed int + float accumulators in same loop
__global__ void mixed_accum(int *out_int, float *out_float, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int isum = 0;
        float fsum = 0.0f;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            isum += v;
            fsum += (float)v;   // implicit cast
        }
        out_int[0] = isum;
        out_float[0] = fsum;
    }
}

// Dot product: sum of element-wise products
__global__ void dot_product(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int dot = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
        }
        out[0] = dot;
    }
}
