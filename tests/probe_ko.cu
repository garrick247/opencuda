// Probe: for(;;) infinite loop with break,
// while(1) infinite loop with break,
// extern __shared__ dynamic shared memory,
// do-while(0) single-iteration wrapper (common C idiom)

// for(;;) infinite loop — classic CAS/retry pattern style
__global__ void infinite_for_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        for (;;) {
            if (i >= n) break;
            sum += in[i];
            i++;
        }
        *out = sum;
    }
}

// while(1) equivalent
__global__ void while_one_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        while (1) {
            if (i >= n) break;
            sum += in[i++];
        }
        *out = sum;
    }
}

// extern __shared__ — dynamic shared memory allocation
__global__ void dynamic_shared(int *out, int *in, int n) {
    extern __shared__ int s[];
    int tid = threadIdx.x;
    if (tid < n) {
        s[tid] = in[tid] * 2;
    }
    __syncthreads();
    if (tid < n) {
        out[tid] = s[tid];
    }
}

// do { ... } while (0) — single-iteration wrapper, always runs once
__global__ void do_while_zero(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        do {
            result = v + 1;
        } while (0);
        out[tid] = result;
    }
}

// Infinite loop used for binary search (find first bit set)
__global__ void find_first_set(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = -1;
        if (v != 0) {
            int i = 0;
            for (;;) {
                if ((v >> i) & 1) {
                    pos = i;
                    break;
                }
                i++;
                if (i >= 32) break;  // safety guard
            }
        }
        out[tid] = pos;
    }
}
