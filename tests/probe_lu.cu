// Probe: trip-count-16 unroll (max boundary), LE loop unroll,
// store-only loop body, nested loop with inner unroll,
// loop with only a function call in body

// Trip count = 16 (max_unroll boundary — should unroll)
__global__ void unroll_16(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 16; i++) {
            sum += i;   // 0+1+...+15 = 120
        }
        out[0] = sum;
    }
}

// Trip count = 17 (above max — should NOT unroll, stays as loop)
__global__ void no_unroll_17(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 17; i++) {
            sum += i;   // 0+1+...+16 = 136
        }
        out[0] = sum;
    }
}

// Store-only loop body (no accumulator)
__global__ void fill_array(int *out, int n, int val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            out[i] = val;
        }
    }
}

// Nested loop: inner has trip 4, outer has trip 3
// Inner should unroll, outer should NOT (trip 3 ≤ 16 but has nested body)
__global__ void nested_unroll(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int r = 0; r < 3; r++) {
            int row_sum = 0;
            for (int c = 0; c < 4; c++) {
                row_sum += in[r * 4 + c];
            }
            total += row_sum;
        }
        out[0] = total;
    }
}

// Loop with only atomicAdd in body
__global__ void atomic_loop(int *counter, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            if (in[i] > 0) atomicAdd(counter, 1);
        }
    }
}
