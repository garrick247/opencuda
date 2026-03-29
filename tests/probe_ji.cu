// Probe: long long / int64_t arithmetic,
// nested loop where inner loop start depends on outer index,
// do-while with continue,
// (int) cast in float accumulation

// 64-bit accumulator summing 32-bit inputs
__global__ void sum_s64(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long sum = 0LL;
        for (int i = 0; i < n; i++) {
            sum += (long long)in[i];
        }
        *out = sum;
    }
}

// Nested loop: inner starts from outer index (upper-triangular pattern)
__global__ void upper_tri_sum(int *out, int *mat, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                total += mat[i * n + j];
            }
        }
        *out = total;
    }
}

// do-while with continue (skip negatives)
__global__ void do_while_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        do {
            if (in[i] < 0) {
                i++;
                continue;
            }
            sum += in[i];
            i++;
        } while (i < n);
        *out = sum;
    }
}

// (int) cast from float in accumulation
__global__ void float_to_int_accum(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += (int)in[i];   // truncating cast each iteration
        }
        *out = sum;
    }
}
