// Probe: optimizer correctness stress — patterns that should NOT be
// folded/hoisted/CSE'd, aliasing through pointers, and unroll + CSE interaction.

// ------------------------------------------------------------------
// Load through pointer in loop — must NOT hoist (alias with stores).

__global__ void ptr_alias_no_hoist(int *out, int *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Reads and writes to buf — LICM must not hoist the load.
        for (int i = 0; i < 4; i++) {
            int v = buf[tid];  // load — aliased with write below
            buf[tid] = v + 1;  // write
        }
        out[tid] = buf[tid];
    }
}

// ------------------------------------------------------------------
// Loop where folding would give wrong answer if done cross-iteration.

__global__ void no_cross_iter_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = 1;
        for (int i = 0; i < 4; i++) {
            x = x * 2 + i;  // carries: x depends on previous iteration's x
        }
        out[tid] = x;
    }
}

// ------------------------------------------------------------------
// Two CSE candidates that are the SAME expression but computed in
// adjacent basic blocks (should NOT be merged across blocks).

__global__ void no_cross_block_cse(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a;
        if (v > 0) {
            a = v * v + 1;  // CSE candidate
        } else {
            a = v * v + 1;  // same expression, different block
        }
        out[tid] = a;
    }
}

// ------------------------------------------------------------------
// Unrolled loop: per-iteration constants should be different.

__global__ void unroll_per_iter_const(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            // Each iteration has a different constant: i*i
            acc += tid + i * i;
        }
        // Expected: tid*4 + (0+1+4+9) = tid*4 + 14
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// CSE across assignments: x*y computed, then x changes, then x*y again.
// The second x*y is NOT the same as the first.

__global__ void cse_after_reassign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int y = x + 1;
        int a = x * y;   // x*y with original x
        x = x + 1;       // x is now different
        int b = x * y;   // x*y with new x — different value, must NOT be CSE'd
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Strength reduction: powers of 2 should become shifts.

__global__ void power_of_two_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = v / 2 + v / 4 + v / 8 + v / 16;
    }
}

// ------------------------------------------------------------------
// Const folding: nested operations all with compile-time operands.

__global__ void deep_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 1 << 8;         // 256
        int b = a >> 4;         // 16
        int c = b * b;          // 256
        int d = c + a;          // 512
        int e = d / b;          // 32
        int f = e * e * e;      // 32768
        out[tid] = f + tid;     // tid varies, but f is constant
    }
}

// ------------------------------------------------------------------
// Dead code after unconditional branch — optimizer must preserve semantics.

__global__ void dead_after_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v > 0) {
            r = v * 2;
            goto done;
            r = 999;  // dead — never reached
        } else {
            r = -v;
        }
    done:
        out[tid] = r;
    }
}
