// Probe: complex loop termination — loop with multiple exit conditions
// via break, loop with && in condition that has side effects in second operand,
// nested loops with shared loop variable

__global__ void complex_loop_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int j = n - 1;
        int sum = 0;

        // Loop that modifies both indices
        while (i < j) {
            sum += in[i] + in[j];
            i++;
            j--;
            if (sum > 1000) break;
        }
        out[tid] = sum + i + j;
    }
}

// Nested loop with shared outer variable modified inside inner
__global__ void nested_shared_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int total = 0;
        int i = 0;
        while (i < n) {
            int j = 0;
            while (j < 4 && i < n) {
                total += in[i];
                i++;
                j++;
            }
        }
        out[tid] = total;
    }
}

// For-loop with complex increment expression
__global__ void complex_increment(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        int i = 0, step = 1;
        for (; i < n; i += step, step = step < 4 ? step + 1 : 1) {
            sum += in[i];
        }
        out[tid] = sum;
    }
}
