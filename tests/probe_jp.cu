// Probe: for-loop re-declaring a name from outer scope,
// sequential for-loops sharing same variable name,
// variable modified in one branch only (merge sees both paths),
// loop variable used after loop with same name as inner-scope variable

// Outer 'i' exists; for-loop declares 'int i' — inner should shadow outer
__global__ void outer_i_shadow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 99;   // outer i
        int sum = 0;
        for (int i = 0; i < n; i++) {  // inner i shadows outer i
            sum += in[i];
        }
        out[0] = sum;
        out[1] = i;  // should be 99, not n
    }
}

// Sequential loops with same loop variable name: second loop must not
// inherit the state from the first loop
__global__ void sequential_loops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum1 = 0, sum2 = 0;
        for (int i = 0; i < n; i++) {
            sum1 += in[i];
        }
        // Second loop: i starts at 0 again
        for (int i = 0; i < n; i++) {
            sum2 += in[n - 1 - i];
        }
        out[0] = sum1;
        out[1] = sum2;
    }
}

// Variable only modified in if-branch, not else: after merge it has
// branch-specific value vs untouched value
__global__ void one_branch_modify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 0;
        if (v > 0) {
            x = v * 2;
            // else: x stays 0
        }
        out[tid] = x;  // should be v*2 or 0
    }
}

// Variable declared before loop, second loop uses a DIFFERENT variable
// with the same name (inner scope redeclaration)
__global__ void reuse_name_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int k = 0; k < n; k++) {
            int val = in[k];  // 'val' declared in inner scope
            total += val;
        }
        // After loop: 'val' is out of scope
        // Now use 'val' as a separate outer variable
        int val = total / n;  // completely separate 'val'
        out[0] = val;
        out[1] = total;
    }
}
