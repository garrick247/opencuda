// Probe: nested loops with struct-updating inlines + shared memory + pred pressure
// Tests outer-loop writeback after inner-loop mutates struct fields, and
// that predicate registers don't alias when many conditions are in flight.

// ------------------------------------------------------------------
// Nested loop struct update: inner loop calls a struct-updating inline,
// outer loop accumulates a second field.  After inner loop exits, the
// outer loop's writeback must see the UPDATED inner-loop state.

struct Stats { float sum; float sumsq; };

__device__ Stats accum(Stats s, float v) {
    s.sum   += v;
    s.sumsq += v * v;
    return s;
}

__global__ void nested_accum(float *out, float *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s; s.sum = 0.0f; s.sumsq = 0.0f;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                s = accum(s, data[idx]);
            }
        }
        out[0] = s.sum;
        out[1] = s.sumsq;
    }
}

// ------------------------------------------------------------------
// Struct update inside nested loop with outer loop carrying a scalar.
// Exercises that the outer loop's scalar writeback doesn't disturb
// the struct's inner-loop writeback and vice versa.

struct MinMax { float mn; float mx; };

__device__ MinMax update_minmax(MinMax m, float v) {
    if (v < m.mn) m.mn = v;
    if (v > m.mx) m.mx = v;
    return m;
}

__global__ void row_minmax(float *out, float *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float row_sum = 0.0f;
        MinMax global_mm; global_mm.mn = data[0]; global_mm.mx = data[0];
        for (int i = 0; i < rows; i++) {
            float row_min = data[i * cols];
            for (int j = 0; j < cols; j++) {
                float v = data[i * cols + j];
                global_mm = update_minmax(global_mm, v);
                if (v < row_min) row_min = v;
            }
            row_sum += row_min;
        }
        out[0] = global_mm.mn;
        out[1] = global_mm.mx;
        out[2] = row_sum;
    }
}

// ------------------------------------------------------------------
// Predicate pressure: a single loop with 4 nested conditions that all
// remain live simultaneously.  Tests that predicate registers p0..p3
// don't alias when conditions are evaluated in sequence.

__global__ void pred_pressure(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v > 1.0f)  a += v;
            if (v > 2.0f)  b += v;
            if (v > 4.0f)  c += v;
            if (v > 8.0f)  d += v;
        }
        out[0] = a; out[1] = b; out[2] = c; out[3] = d;
    }
}
