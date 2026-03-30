// Probe: correctness audit — patterns where incorrect codegen would produce
// valid PTX that computes wrong results. These test semantic correctness
// of operator precedence, short-circuit evaluation, integer promotion,
// and loop-carried phi nodes.

// ------------------------------------------------------------------
// Operator precedence: & vs == (& binds tighter than == in C).

__global__ void precedence_and_eq(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // C precedence: (v & 1) == 0, NOT v & (1 == 0)
        out[tid] = (v & 1) == 0 ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Shift vs add precedence: << binds tighter than +.

__global__ void precedence_shift_add(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // C: (v << 2) + 1, NOT v << (2 + 1)
        out[tid] = v << 2 + 1;  // This is actually v << (2+1) = v << 3! (+ binds tighter than <<)
        // Wait, C precedence: + is HIGHER than <<
        // So v << 2 + 1 = v << (2 + 1) = v << 3
    }
}

// ------------------------------------------------------------------
// Ternary associativity: right-to-left.

__global__ void ternary_assoc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // a ? b : c ? d : e  ≡  a ? b : (c ? d : e)  (right-to-left)
        out[tid] = v > 0 ? 1 : v == 0 ? 0 : -1;
    }
}

// ------------------------------------------------------------------
// Integer promotion: char + char should promote to int.

__global__ void char_promotion(int *out, signed char *a, signed char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Both chars promote to int before addition
        // If a=100, b=100, sum should be 200 (not overflow to -56)
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// Short-circuit &&: second operand must not execute if first is false.

__global__ void short_circuit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = in[tid];
        // If idx < 0 || idx >= n, the second condition must not be evaluated
        // (would be out-of-bounds access)
        int safe = (idx >= 0 && idx < n) ? in[idx] : -1;
        out[tid] = safe;
    }
}

// ------------------------------------------------------------------
// Post-increment in loop condition.

__global__ void postinc_condition(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int i = 0;
        // Post-increment in condition: use current i, then increment
        while (i < 8) {
            s += in[(tid + i) % n];
            i++;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Loop-carried value modified in only one branch of an if.

__global__ void partial_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        int last_pos = -1;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            s += v;
            if (v > 0) last_pos = i;  // only updated sometimes
        }
        out[tid * 2]     = s;
        out[tid * 2 + 1] = last_pos;
    }
}

// ------------------------------------------------------------------
// Comma operator in for-loop init and update.

__global__ void comma_all(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0, j = tid; i < j; i++, j--) {
            s += i * j;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Assignment as expression: x = y = z = 0.

__global__ void chain_assign(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x, y, z;
        x = y = z = tid + 1;
        out[tid] = x + y + z;  // should be 3 * (tid + 1)
    }
}

// ------------------------------------------------------------------
// Unsigned right shift vs signed right shift.

__global__ void shift_signedness(int *out_arith, unsigned *out_logic,
                                    int *in_s, unsigned *in_u, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Signed right shift: arithmetic (sign-extends)
        out_arith[tid] = in_s[tid] >> 4;
        // Unsigned right shift: logical (zero-fills)
        out_logic[tid] = in_u[tid] >> 4;
    }
}
