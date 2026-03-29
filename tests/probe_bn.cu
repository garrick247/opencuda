// Probe: Unusual loop patterns and loop nesting
// - Infinite for loop with internal break on ALL paths
// - Loop variable used after loop
// - Loop variable modified in both body and condition
// - Nested while with index math
// - For loop with non-trivial init expression

__global__ void countdown_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int count = n;
        while (count > 0) {
            count--;
        }
        out[tid] = count;  // should be 0
    }
}

__global__ void loop_var_after(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i;
        for (i = 0; i < n && i < 16; i++) {
            // do nothing
        }
        out[tid] = i;  // i = min(n, 16)
    }
}

__global__ void nested_while_math(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0, j = 0, sum = 0;
        while (i < n) {
            j = i;
            while (j < n && j < i + 4) {
                sum += j;
                j++;
            }
            i += 4;
        }
        out[tid] = sum + tid;
    }
}
