// Probe: loop with LE condition unrolled correctly,
// loop with stride != 1 (stride 2, trip 5),
// for(;;) with early break,
// double-precision accumulator,
// loop body calling __syncthreads (should NOT unroll or handle correctly)

// Loop with i <= N: for(i=0; i<=3; i++) — trip 4, sum = 0+1+2+3 = 6
__global__ void loop_le_small(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i <= 3; i++) {
            sum += i;
        }
        out[0] = sum;   // 6
    }
}

// Stride-2 loop: for(i=0; i<10; i+=2) — loads in[0,2,4,6,8]
__global__ void stride2_sum(int *out, int *in) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 10; i += 2) {
            sum += in[i];
        }
        out[0] = sum;
    }
}

// for(;;) with break: runs exactly 4 times then exits
__global__ void bounded_infinite(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int i = 0;
        for (;;) {
            if (i >= 4) break;
            count++;
            i++;
        }
        out[0] = count;   // 4
    }
}

// Double-precision accumulator loop
__global__ void double_sum(double *out, double *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += in[i];
        }
        out[0] = acc;
    }
}

// Loop with __syncthreads in body — should emit barrier each iteration
// (loop should NOT be unrolled by current unroller since body doesn't
// match the simple writeback pattern)
__global__ void sync_loop(int *shared_arr, int n) {
    int tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        shared_arr[tid] += i;
        __syncthreads();
    }
}
