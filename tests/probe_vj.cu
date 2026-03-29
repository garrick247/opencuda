// Probe: chained assignment, loop with <= bound and non-zero start,
// local array initialization, negative-step loop,
// and multi-level constant expression chains.

// ------------------------------------------------------------------
// Chained assignment: x = y = z = 0.

__global__ void chained_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x, y, z;
        x = y = z = in[tid];
        out[tid] = x + y + z;  // 3 * in[tid]
    }
}

// ------------------------------------------------------------------
// Loop with <= bound: for(i=0; i<=4; i++).

__global__ void le_bound_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i <= 4; i++) {  // 5 iterations: 0,1,2,3,4
            acc += v + i;
        }
        out[tid] = acc;  // 5v + 10
    }
}

// ------------------------------------------------------------------
// Loop starting at non-zero initial value: for(i=2; i<6; i++).

__global__ void nonzero_start_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 2; i < 6; i++) {  // 4 iterations: 2,3,4,5
            acc += v + i;
        }
        out[tid] = acc;  // 4v + 14
    }
}

// ------------------------------------------------------------------
// Countdown loop: for(i=4; i>0; i--).

__global__ void countdown_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 4; i > 0; i--) {  // 4 iterations: 4,3,2,1
            acc += v * i;
        }
        out[tid] = acc;  // v * (4+3+2+1) = 10v
    }
}

// ------------------------------------------------------------------
// Local array with full initializer list.

__global__ void local_array_init(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int lut[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        int idx = v & 7;  // v mod 8
        out[tid] = lut[idx];
    }
}

// ------------------------------------------------------------------
// Multi-level constant arithmetic (tests chain fold improvement).

__global__ void deep_const_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // 6-level chain — previously would stop early
        int a = 1 + 2;       // 3
        int b = a + 3;       // 6
        int c = b + 4;       // 10
        int d = c + 5;       // 15
        int e = d + 6;       // 21
        int f = e + 7;       // 28
        out[tid] = v + f;    // v + 28
    }
}

// ------------------------------------------------------------------
// Negative-step loop: while(n > 0) n -= 3.

__global__ void neg_step_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int cnt = 9;
        while (cnt > 0) {
            acc += v;
            cnt -= 3;  // 3 iterations: cnt=9,6,3
        }
        out[tid] = acc;  // 3v
    }
}

// ------------------------------------------------------------------
// Loop with >= bound: for(i=10; i>=6; i--) — 5 iterations.

__global__ void ge_bound_countdown(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 10; i >= 6; i--) {  // 5 iters: 10,9,8,7,6
            acc += v + i;
        }
        out[tid] = acc;  // 5v + 40
    }
}
