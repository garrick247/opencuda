// Probe: multiple variables updated in both if and else branches (phi merge),
// loop with multiple accumulators updated conditionally,
// early exit from nested loop (flag variable break pattern),
// while loop that modifies multiple variables per iteration

// Both if and else update two variables
__global__ void branch_two_vars(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = 0, neg = 0;
        if (v > 0) {
            pos = v;
            neg = 0;
        } else {
            pos = 0;
            neg = -v;
        }
        out[tid * 2]     = pos;
        out[tid * 2 + 1] = neg;
    }
}

// Loop with two conditionally updated accumulators
__global__ void conditional_two_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_sum = 0, neg_sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v >= 0) {
                pos_sum += v;
            } else {
                neg_sum += (-v);
            }
        }
        out[0] = pos_sum;
        out[1] = neg_sum;
    }
}

// Nested loop early exit via flag variable
__global__ void find_pair(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found_i = -1, found_j = -1;
        int done = 0;
        for (int i = 0; i < n && !done; i++) {
            for (int j = i + 1; j < n && !done; j++) {
                if (a[i] + b[j] == 0) {
                    found_i = i;
                    found_j = j;
                    done = 1;
                }
            }
        }
        out[0] = found_i;
        out[1] = found_j;
    }
}

// While loop updating three variables simultaneously
__global__ void gcd_loop(int *out, int a_val, int b_val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = a_val, b = b_val;
        int steps = 0;
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
            steps++;
        }
        out[0] = a;
        out[1] = steps;
    }
}
