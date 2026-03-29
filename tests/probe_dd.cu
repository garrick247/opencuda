// Probe: Pattern that can expose issues in the for-loop writeback mechanism
// - Loop where condition variable is modified outside the loop header
// - Loop where the increment is a complex expression
// - Loop where break is conditional on a non-trivial expression
// - For loop with missing init (skipped): for (; cond; inc)

__global__ void complex_increment(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int i = 0;
        for (; i < n; i = i + 3) {  // increment by 3 in update expression
            sum += i;
        }
        out[tid] = sum;
    }
}

__global__ void conditional_break_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            if (sum > 1000) break;  // break when sum exceeds threshold
        }
        out[tid] = sum;
    }
}

// Loop with multiple exit conditions
__global__ void multi_exit_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int found = -1;
        while (i < n && found < 0) {
            if (in[i] == tid) {
                found = i;
            }
            i++;
        }
        out[tid] = found;
    }
}
