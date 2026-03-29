// Probe: nested if inside loop with &&, complex break conditions,
// and pre-decrement in loop update.

// ------------------------------------------------------------------
// Compound if inside a loop with guarded array access.
// `if (i > 0 && data[i-1] > data[i])` — tests short-circuit in loop body.

__global__ void inner_compound_if(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int inversions = 0;
        for (int i = 0; i < n; i++) {
            if (i > 0 && data[i-1] > data[i]) {
                inversions++;
            }
        }
        out[0] = inversions;
    }
}

// ------------------------------------------------------------------
// Break with compound condition: break when both conditions hold.

__global__ void break_on_compound(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] < 0 && sum > 100) break;
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Pre-decrement in loop update: for (int i = n-1; i >= 0; --i).
// Tests that --i in for increment is handled (not i-- postfix).

__global__ void pre_decrement_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = n - 1; i >= 0; --i) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Logical NOT on compound expression: !(a && b).
// Tests that ! correctly inverts the short-circuit result.

__global__ void not_and(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int result = !(v > 0 && v < 100);   // NOT (positive AND < 100)
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Compound ||: skip load if first condition is true.
// `data[tid] > 0 || alt[tid] > 0` — alt should not be loaded if data is positive.

__global__ void or_guard(int *out, int *data, int *alt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (data[tid] > 0 || alt[tid] > 0) ? 1 : 0;
    }
}
