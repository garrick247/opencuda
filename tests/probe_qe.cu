// Probe: partial liveness / conditional-define patterns.
// Values defined on only one branch of a conditional inside a loop,
// used after the loop. Stresses back-edge liveness and phi placement.

// ------------------------------------------------------------------
// Variable defined only in if-branch, used after loop.
// `last_pos` is only updated when v > 0; used after loop.

__global__ void last_positive(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int last_pos = -1;
        for (int i = 0; i < n; i++) {
            if (data[i] > 0) last_pos = i;
        }
        out[0] = last_pos;
    }
}

// ------------------------------------------------------------------
// Two variables each updated on a different branch.
// `a` updated in if-branch, `b` in else-branch. Both used after loop.

__global__ void split_branch_vars(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_max = 0, neg_min = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            if (v > pos_max) pos_max = v;
            else if (v < neg_min) neg_min = v;
        }
        out[0] = pos_max - neg_min;
    }
}

// ------------------------------------------------------------------
// Loop with flag set once, read after loop.
// `found` set in first match only; must remain live after loop.

__global__ void find_first_flag(int *out, int *data, int n, int target) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found = 0;
        int found_idx = -1;
        for (int i = 0; i < n; i++) {
            if (!found && data[i] == target) {
                found = 1;
                found_idx = i;
            }
        }
        out[0] = found_idx;
    }
}

// ------------------------------------------------------------------
// Conditional accumulation: two separate sums for even/odd indices.
// Both `even_sum` and `odd_sum` are partially updated.

__global__ void even_odd_sum(int *out_even, int *out_odd, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int even_sum = 0, odd_sum = 0;
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) even_sum += data[i];
            else            odd_sum  += data[i];
        }
        out_even[0] = even_sum;
        out_odd[0]  = odd_sum;
    }
}

// ------------------------------------------------------------------
// Variable that escapes a nested conditional.
// `best` updated inside if/else-if/else chain.

__global__ void nested_cond_escape(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int best = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            if      (v > 100) best = v;
            else if (v > 50)  best = (best < v) ? v : best;
            else if (v > 10)  best += 1;
            // else: best unchanged
        }
        out[0] = best;
    }
}

// ------------------------------------------------------------------
// Loop-carried pair: (sum, count) both updated only in if-branch.

__global__ void cond_pair(int *out_sum, int *out_cnt, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0, count = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            if (v != 0) {
                sum   += v;
                count += 1;
            }
        }
        out_sum[0] = sum;
        out_cnt[0] = count;
    }
}

// ------------------------------------------------------------------
// Nested loop: outer loop-carried value modified in inner loop only.
// `prefix` accumulates in inner, used in outer after inner exits.

__global__ void nested_partial(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < rows; i++) {
            int row_max = 0;
            for (int j = 0; j < cols; j++) {
                int v = data[i * cols + j];
                if (v > row_max) row_max = v;
            }
            out[i] = row_max;
        }
    }
}

// ------------------------------------------------------------------
// Conditional assignment to loop-carried var in while loop.
// `cur_min` changes only on the taken branch.

__global__ void while_cond_update(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int cur_min = 2147483647;
        while (i < n) {
            int v = data[i];
            if (v < cur_min) cur_min = v;
            i++;
        }
        out[0] = cur_min;
    }
}
