// Probe: predicate combining, boolean-to-int conversions,
// conditional compound ops, and comparison result arithmetic.

// ------------------------------------------------------------------
// Boolean-to-int: comparison result used in arithmetic.

__global__ void cmp_as_int(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Comparisons used directly in arithmetic (implicit int cast)
        int a = (v > 0);         // 0 or 1
        int b = (v < 100);       // 0 or 1
        int c = (v == 42);       // 0 or 1
        int d = (v != 0);        // 0 or 1
        out[tid] = a + b * 2 + c * 4 + d * 8;
    }
}

// ------------------------------------------------------------------
// Conditional compound: v += (a > b) ? x : y.

__global__ void cond_compound(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        // Each += uses a conditional value
        acc += (v > 0)  ? v     : -v;
        acc += (v < 10) ? v * 2 : v + 1;
        acc += (v == 5) ? 100   : 0;
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Predicate AND: (a > 0 && b < 10) — both conditions in one branch.

__global__ void pred_and(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid];
        int bv = b[tid];
        int r = 0;
        // Short-circuit AND: if first is false, skip second
        if (av > 0 && bv < 10) r = 1;
        if (av > 5 && bv < 5)  r += 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate OR: combined conditions.

__global__ void pred_or(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid];
        int bv = b[tid];
        int r = 0;
        if (av < 0 || bv > 100) r = 1;
        if (av > 50 || bv < -50) r += 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Comparison result stored and reused.

__global__ void stored_predicate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = (v > 0);         // store comparison as int
        int large = (v > 1000);
        int both = pos & large;    // bitwise AND on comparison results
        int either = pos | large;  // bitwise OR
        out[tid] = both + either * 2 + pos * 4 + large * 8;
    }
}

// ------------------------------------------------------------------
// Multiply boolean flags to accumulate weights.

__global__ void bool_weight_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float score = 0.0f;
        // Each condition contributes a weight
        score += (v > 0.0f)   ? 1.0f : 0.0f;
        score += (v > 1.0f)   ? 1.0f : 0.0f;
        score += (v > 10.0f)  ? 1.0f : 0.0f;
        score += (v > 100.0f) ? 1.0f : 0.0f;
        out[tid] = score;
    }
}

// ------------------------------------------------------------------
// Comparison feeding into loop counter (accumulate when condition true).

__global__ void cond_count(int *out, int *in, int n, int threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        for (int i = 0; i < 8; i++) {
            // Conditional increment: count += (shifted v's ith byte > threshold)
            int byte = (v >> (i * 4)) & 0xF;
            count += (byte > threshold) ? 1 : 0;
        }
        out[tid] = count;
    }
}
