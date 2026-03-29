// Probe: secondary induction correctness, loop-carried float,
// interleaved multi-variable loops, and boundary conditions.

// ------------------------------------------------------------------
// Two counter variables, one up one down (same as for_comma_update
// but with a different accumulation to verify both are correct).

__global__ void two_counter_sum(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        // i goes 0→4, j goes 8→4 (step -1).
        // i * j = 0*8, 1*7, 2*6, 3*5, 4*4 = 0+7+12+15+16 = 50
        for (int i = 0, j = 8; i < 5; i++, j--) {
            acc += i * j;
        }
        out[tid] = acc;  // expected: 50
    }
}

// ------------------------------------------------------------------
// Three simultaneous counters: i++, j-=2, k*= (not detectable — k is not linear).

__global__ void two_linear_counters(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        // i: 0→3 step +1, j: 10→4 step -2
        // (i+j): 10, 10, 10, 10 (each step i+1, j-2, net change = -1)
        // Wait: i=0,j=10 → 10; i=1,j=8 → 9; i=2,j=6 → 8; i=3,j=4 → 7
        // Sum = 34
        for (int i = 0, j = 10; i < 4; i++, j -= 2) {
            acc += i + j;
        }
        out[tid] = acc;  // expected: 10+9+8+7 = 34
    }
}

// ------------------------------------------------------------------
// Loop with float secondary induction (e.g. step = +0.5f, but floats
// won't constant-fold — test that the loop still produces correct output).

__global__ void loop_float_secondary(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float acc = 0.0f;
        // i is int induction, weight is float but NOT secondary (runtime val)
        for (int i = 0; i < 4; i++) {
            acc += v * (float)i;
        }
        out[tid] = acc;  // = v * (0+1+2+3) = 6*v
    }
}

// ------------------------------------------------------------------
// Single unrolled loop with accumulation and post-loop use.

__global__ void unroll_accumulate_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        // Trip count = 8, should unroll
        for (int i = 0; i < 8; i++) {
            acc += v + i;
        }
        // acc = 8*v + 28
        out[tid] = acc - tid;
    }
}

// ------------------------------------------------------------------
// Loop with secondary counter used in address calculation.

__global__ void secondary_in_addr(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        // i goes forward, j goes backward through arrays
        // Reading a[i] and b[j] simultaneously
        int len = 4;
        for (int i = 0, j = len - 1; i < len; i++, j--) {
            acc += a[i] + b[j];
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop that walks two arrays in same direction with different strides.
// j = i * 3 (not linear step — just different base/stride).

__global__ void dual_stride_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            // Access two different elements per iteration
            acc += in[tid * 8 + i] + in[tid * 8 + i + 4];
        }
        out[tid] = acc;
    }
}
