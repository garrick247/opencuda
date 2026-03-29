// Probe: __shared__ memory loaded from global then used in computation,
// long long accumulation (64-bit prefix-sum style),
// switch-break inside a for-loop (break exits switch, not loop),
// multiple __shared__ arrays in same kernel

// Two __shared__ arrays: one for data, one for flags
__global__ void shared_two_arrays(int *out, int *data, int *flags, int n) {
    __shared__ int s_data[32];
    __shared__ int s_flag[32];
    int tid = threadIdx.x;
    if (tid < n) {
        s_data[tid] = data[tid];
        s_flag[tid] = flags[tid];
    }
    __syncthreads();
    if (tid < n) {
        // Write back: data * flag (flag is 0 or 1)
        out[tid] = s_data[tid] * s_flag[tid];
    }
}

// Long long accumulation: sum of squares may overflow int32
__global__ void sum_squares_ll(long long *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long total = 0LL;
        for (int i = 0; i < n; i++) {
            long long v = (long long)in[i];
            total += v * v;
        }
        *out = total;
    }
}

// switch inside for-loop: break exits switch, loop continues
__global__ void switch_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int cnt0 = 0, cnt1 = 0, cnt2 = 0, cntx = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i] % 4;   // 0, 1, 2, or 3
            switch (v) {
                case 0: cnt0++; break;
                case 1: cnt1++; break;
                case 2: cnt2++; break;
                default: cntx++; break;
            }
            // loop continues after switch regardless of which case ran
        }
        out[0] = cnt0;
        out[1] = cnt1;
        out[2] = cnt2;
        out[3] = cntx;
    }
}

// __shared__ reduction: each thread loads one element, then
// sequential reduce by thread 0 using shared memory
__global__ void shared_reduce(int *out, int *in, int n) {
    __shared__ int s[32];
    int tid = threadIdx.x;
    if (tid < n) {
        s[tid] = in[tid];
    } else {
        s[tid] = 0;
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += s[i];
        }
        *out = sum;
    }
}
