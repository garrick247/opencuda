// Probe: multi-variable for-loop init (int i=0, step=1),
// break and continue scope interactions,
// variable modified after break/continue (writeback correctness),
// for-loop with both outer and inner use of variable names

// Multi-init for-loop: both i and step are loop-scoped
__global__ void multi_init_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int step = 999;   // outer step, should survive the loop
        for (int i = 0, s = 1; i < n; i += s) {
            sum += in[i];
            if (i > n / 2) s = 2;   // change inner step variable
        }
        out[0] = sum;
        out[1] = step;   // should still be 999
    }
}

// Break with variable modification before break
__global__ void break_with_mod(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int found = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            sum += v;
            if (v < 0) {
                found = 1;
                break;   // sum and found should persist after break
            }
        }
        out[0] = sum;
        out[1] = found;
    }
}

// Continue in nested if: variable before continue must writeback
__global__ void continue_nested_if(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_count = 0, neg_count = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v == 0) continue;
            if (v > 0) {
                pos_count++;
                continue;
            }
            neg_count++;
        }
        out[0] = pos_count;
        out[1] = neg_count;
    }
}

// Outer and inner loops both named i — break/continue must be innermost
__global__ void nested_same_name_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            for (int i = 0; i < n; i++) {  // inner i shadows outer
                total += in[i];
                if (in[i] < 0) break;   // break inner only
            }
            total += 1;  // executed once per outer iteration
        }
        *out = total;
    }
}
