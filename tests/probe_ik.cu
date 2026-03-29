// Probe: tricky loop patterns — loop with multiple update expressions,
// for-loop with empty body, range-based accumulation,
// loop counter used after loop exit

// Multiple updates in for-loop increment
__global__ void multi_update_for(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = n;
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            a += i;
            b -= j;
        }
        out[0] = a;
        out[1] = b;
    }
}

// Loop counter used after loop body
__global__ void count_until(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i;
        for (i = 0; i < n; i++) {
            if (in[i] < 0) break;
        }
        // i is the index where negative was found, or n if not found
        *out = i;
    }
}

// Accumulate with step > 1
__global__ void stride_sum(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long sum = 0;
        for (int i = 0; i < n; i += 3) {
            sum += in[i];
        }
        *out = sum;
    }
}

// While loop with pre-computed limit
__global__ void while_limit(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mid = n / 2;
        int i = 0;
        float sum = 0.0f;
        while (i < mid) {
            sum += in[i] - in[n - 1 - i];
            i++;
        }
        *out = sum;
    }
}
