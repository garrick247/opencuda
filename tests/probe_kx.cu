// Probe: optimizer interaction edge cases —
// LICM with side-effecting conditional inside loop,
// CSE across two phi-merged branches,
// identity_fold of loop writeback with zero init,
// constant folding chains with sub/neg

// LICM: invariant computation mixed with conditional store
// The invariant `k = a * b` should be hoisted; the store must NOT be
__global__ void licm_cond_store(int *out, int *in, int a, int b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int k = a * b;    // invariant — hoistable
        for (int i = 0; i < n; i++) {
            if (in[i] > 0) {
                out[i] = in[i] + k;   // conditional store — NOT hoistable
            }
        }
    }
}

// identity_fold: loop variable initialized to zero, incremented by 1 each iter
// After unrolling (trip count <= 16) the phi/writeback chain should fully fold
__global__ void identity_fold_counter(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 8; i++) {   // trip count 8 — will be unrolled
            sum += i;
        }
        out[0] = sum;   // should fold to 28
    }
}

// Subtraction constant chain: a-b-c should fold correctly
__global__ void const_sub_chain(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 100;
        int b = 37;
        int c = 24;
        int r1 = a - b;        // 63
        int r2 = r1 - c;       // 39
        int r3 = r2 * 2;       // 78
        int r4 = r3 - r1;      // 15
        out[0] = r4;           // should be 15
    }
}

// Negation and subtraction: -(a - b) vs b - a
__global__ void neg_sub(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int neg_a = -a;
        int b = a - neg_a;     // a - (-a) = 2a
        int c = neg_a + a;     // 0
        out[tid * 2]     = b;
        out[tid * 2 + 1] = c;
    }
}

// Two nested loops, inner has a break — outer loop-carried var must be correct
__global__ void nested_break_outer(int *out, int *in, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int r = 0; r < rows; r++) {
            int row_sum = 0;
            for (int c = 0; c < cols; c++) {
                int v = in[r * cols + c];
                if (v < 0) break;    // break inner loop
                row_sum += v;
            }
            total += row_sum;        // outer accumulates all row sums
        }
        out[0] = total;
    }
}
