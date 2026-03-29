// Probe: do-while, switch with fallthrough, continue in nested loop,
// break in switch inside loop, ternary nested in expression

// do-while: runs at least once, count should be 5
__global__ void do_while_count(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int i = 0;
        do {
            count++;
            i++;
        } while (i < n);
        out[0] = count;   // = n (for n>0), = 1 (for n<=0)
    }
}

// switch with multiple cases falling through to same block
__global__ void switch_classify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int label;
        switch (x % 4) {
            case 0: label = 10; break;
            case 1: label = 20; break;
            case 2: label = 30; break;
            default: label = 40; break;
        }
        out[tid] = label;
    }
}

// continue inside inner loop — skips even indices
__global__ void skip_evens(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) continue;   // skip even indices
            sum += in[i];
        }
        out[0] = sum;   // sum of in[1] + in[3] + in[5] + ...
    }
}

// break in switch inside a loop
__global__ void categorized_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_sum = 0, neg_sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            int category;
            switch (v > 0 ? 1 : (v < 0 ? -1 : 0)) {
                case  1: pos_sum += v; break;
                case -1: neg_sum += v; break;
                default: break;
            }
        }
        out[0] = pos_sum;
        out[1] = neg_sum;
    }
}

// Ternary nested in arithmetic expression
__global__ void clamped_scale(int *out, int *in, int n, int lo, int hi) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // clamp to [lo, hi] then double
        int clamped = (v < lo) ? lo : ((v > hi) ? hi : v);
        out[tid] = clamped * 2;
    }
}
