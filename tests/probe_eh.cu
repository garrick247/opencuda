// Probe: Patterns that might expose issues with variable tracking across
//        complex control flow (if-in-loop, loop-in-if, etc.)

__global__ void if_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int neg = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) {
                sum += v;
            } else {
                neg++;
            }
        }
        out[tid] = sum - neg;
    }
}

// Nested if in nested loop
__global__ void nested_if_loop(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (a[i] + a[j] == b[tid]) {
                    count++;
                }
            }
        }
        out[tid] = count;
    }
}

// Loop with conditional that modifies multiple vars
__global__ void multi_update_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum_pos = 0;
        int sum_neg = 0;
        int count_zero = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) { sum_pos += v; }
            else if (v < 0) { sum_neg += v; }
            else { count_zero++; }
        }
        out[tid] = sum_pos + sum_neg + count_zero;
    }
}
