// Probe: continue skips side-effect code (loop writeback must still be correct),
// decrement-in-condition (--i >= 0 style),
// pointer passed to device fn (array pointer + offset),
// variable both modified before and after continue in same iteration,
// flag variable toggled inside loop with if/else

// continue skips code that would have modified 'total'
__global__ void continue_skip_code(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        int skipped = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] == 0) {
                skipped++;
                continue;      // skip the total += in[i] below
            }
            total += in[i];   // only runs if in[i] != 0
        }
        out[0] = total;
        out[1] = skipped;
    }
}

// Decrement in condition: for (int i = n; --i >= 0;)
__global__ void decrement_in_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = n; --i >= 0;) {  // decrement happens in condition
            sum += in[i];
        }
        *out = sum;
    }
}

// Device function receives pointer + offset, accesses array
__device__ int window3_sum(int *arr, int offset) {
    return arr[offset] + arr[offset + 1] + arr[offset + 2];
}

__global__ void sliding_window(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n - 2) {
        out[tid] = window3_sum(in, tid);
    }
}

// Variable modified before and after continue: 'partial' updates before,
// 'total' only updates if not skipped
__global__ void partial_before_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        int partial = 0;
        for (int i = 0; i < n; i++) {
            partial += 1;          // always increments (even for skipped)
            if (in[i] < 0) continue;
            total += in[i];        // only for non-negative
        }
        out[0] = total;
        out[1] = partial;   // must equal n (all iterations increment partial)
    }
}

// Flag toggled inside if/else inside loop
__global__ void toggle_flag(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int flag = 0;
        int transitions = 0;
        int prev = flag;
        for (int i = 0; i < n; i++) {
            prev = flag;
            if (in[i] > 0) {
                flag = 1;
            } else {
                flag = 0;
            }
            if (flag != prev) transitions++;
        }
        out[0] = flag;
        out[1] = transitions;
    }
}
