// Probe: if-else where one branch continues and the other falls through,
// variables modified before an early return in a device function,
// if-else where else has a nested loop that modifies outer variable,
// multiple variables modified in asymmetric branches

// Loop: if continues, else modifies and falls through
__global__ void asymmetric_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int skipped = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v == 0) {
                skipped++;
                continue;   // skip accumulation, go to next i
            } else {
                sum += v;
                // no continue: falls through to end of loop body
            }
        }
        out[0] = sum;
        out[1] = skipped;
    }
}

// Multiple variables modified asymmetrically: one branch sets A+B, other sets only A
__global__ void partial_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = 0;
        if (v > 0) {
            a = v;
            b = v * 2;
        } else {
            a = -v;
            // b stays 0
        }
        out[tid * 2]     = a;
        out[tid * 2 + 1] = b;
    }
}

// Device function with early return in one branch
__device__ int safe_div(int num, int den) {
    if (den == 0) return 0;   // early return
    return num / den;
}

__global__ void early_return_device(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_div(a[tid], b[tid]);
    }
}

// Else branch has inner loop that modifies outer variable
__global__ void else_inner_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > 0) {
                total += in[i];
            } else {
                // Else: sum all elements up to this point again (weird but tests merge)
                int inner_sum = 0;
                for (int j = 0; j <= i; j++) {
                    inner_sum += in[j] > 0 ? in[j] : 0;
                }
                total -= inner_sum;
            }
        }
        *out = total;
    }
}
