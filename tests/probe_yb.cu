// Probe: Optimizer correctness stress — loop-carried values that survive
// CSE boundaries, values modified in both branches of an if inside a loop,
// loop-carried double accumulator, nested loop where inner loop modifies
// outer-loop variable, constant propagation through chain, loop with
// multiple exits (break in if + natural exit), and address-taken local
// variable modified through pointer inside a loop.

// ------------------------------------------------------------------
// Loop-carried bool flag: found/pos pattern with nested if.

__global__ void loop_flag(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int pos = -1;
        int val = 0;
        for (int i = 0; i < n; i++) {
            int x = in[i];
            if (x > 0 && pos < 0) {
                pos = i;
                val = x;
            }
        }
        out[tid * 2]     = pos;
        out[tid * 2 + 1] = val;
    }
}

// ------------------------------------------------------------------
// Inner loop modifies outer-loop variable (via conditional decrement).

__global__ void outer_var_modify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int total = 0;
        int limit = in[tid];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                int v = in[(i * 4 + j) % n];
                total += v;
                if (v < 0) limit--;  // outer-loop-carried var modified in inner
            }
            if (limit <= 0) break;
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Double accumulator loop: two loop-carried doubles.

__global__ void double_accum(double *out_sum, double *out_sq,
                               double *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        double sum = 0.0;
        double sq  = 0.0;
        for (int i = 0; i < n; i++) {
            double v = in[i];
            sum += v;
            sq  += v * v;
        }
        out_sum[tid] = sum;
        out_sq[tid]  = sq;
    }
}

// ------------------------------------------------------------------
// CSE: same subexpression in both branches inside a loop.

__global__ void cse_in_loop(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < n; i++) {
            int x = a[i] + b[i];  // CSE candidate across loop
            if (x > 0) s += x;
            else       s -= x;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Constant chain propagation: a series of dependent constants.

__global__ void const_chain(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 3;
        int b = a * 2;
        int c = b + a;
        int d = c * b - a;
        int e = d + c;
        out[tid] = e + tid;  // e is fully constant; only tid is runtime
    }
}

// ------------------------------------------------------------------
// Loop with multiple exits.

__global__ void multi_exit(int *out, int *in, int maxval, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int i;
        for (i = 0; i < n; i++) {
            int v = in[i];
            if (v > maxval) break;   // exit 1: value too large
            if (v < 0)      break;   // exit 2: negative value
            s += v;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Address-taken local, modified through pointer inside a loop.

__global__ void ptr_loop_modify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        int *p = &acc;
        for (int i = 0; i < 8; i++) {
            *p += in[(tid + i) % n];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop-carried float: running min/max.

__global__ void running_minmax(float *out_min, float *out_max,
                                  float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float lo = in[0];
        float hi = in[0];
        for (int i = 1; i < n; i++) {
            float v = in[i];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        out_min[tid] = lo;
        out_max[tid] = hi;
    }
}

// ------------------------------------------------------------------
// Nested condition: &&, ||, and ternary all in one expression.

__global__ void complex_expr(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid], y = b[tid], z = c[tid];
        int r = (x != 0 && y != 0) ? (x > y ? x : y) :
                (x == 0 || y == 0) ? z : 0;
        out[tid] = r;
    }
}
