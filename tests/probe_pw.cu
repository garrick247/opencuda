// Probe: optimizer correctness — LICM edge cases, CSE across types,
// constant folding with negative values, dead code after return.

// ------------------------------------------------------------------
// LICM safety: expression uses loop-carried variable (not invariant).
// `a * i` must NOT be hoisted — `i` changes each iteration.

__global__ void licm_not_invariant(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = data[0];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a * i;   // a is invariant, i is not — product is not
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// LICM safety: function call in loop condition side-effect.
// Load from array in condition must not be hoisted.

__global__ void licm_load_in_cond(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] != 0) {   // load must execute each iteration
                sum += data[i];
            }
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Truly loop-invariant expression: should be hoisted.
// `a + b` is invariant (a, b not modified in loop).

__global__ void licm_truly_invariant(int *out, int *data, int n, int a, int b) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int c = a + b;   // computed once outside loop
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i] * c;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// CSE: same address computed twice, different use contexts.
// Both loads from `data[tid*4]` should share the computed address.

__global__ void cse_same_addr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int off = tid * 4;
        int v1 = data[off];         // CSE: off computed once
        int v2 = data[off] + 1;     // same off — should reuse
        out[tid] = v1 + v2;
    }
}

// ------------------------------------------------------------------
// CSE across int and float: `(float)tid` and `tid` must NOT be merged.
// Different types with same operand should be distinct.

__global__ void cse_int_float_separate(float *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   iv = data[tid] + tid;         // int arithmetic
        float fv = (float)data[tid] + (float)tid;  // float arithmetic
        out[tid] = fv + (float)iv;
    }
}

// ------------------------------------------------------------------
// Constant folding: negative constant arithmetic.
// -1 * -1 = 1, -2 + -3 = -5, etc.

__global__ void const_fold_negative(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // These should constant-fold
        int a = v + (-1 + -2);    // v + (-3)
        int b = v * (-1 * -1);    // v * 1 → v
        int c = v - (-4 + -5);    // v - (-9) = v + 9
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Dead code: code after return should not crash the compiler.
// (Dead block elimination should remove unreachable code.)

__global__ void dead_after_return(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid];
        return;
        // Dead: this should never execute
        out[tid] = -999;
    }
}

// ------------------------------------------------------------------
// Identity fold chain: multiple identities in sequence.
// v + 0 → v, v * 1 → v, v ^ 0 → v, v | 0 → v, v & ~0 (skipped), v - 0 → v.

__global__ void identity_chain(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        v = v + 0;     // identity
        v = v * 1;     // identity
        v = v ^ 0;     // identity
        v = v | 0;     // identity
        v = v - 0;     // identity
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// CSE with commutative operation: a+b and b+a should share.
// Tests that commutative normalization enables CSE.

__global__ void cse_commutative(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid];
        int x = va + vb;  // CSE key: (ADD, min_id, max_id)
        int y = vb + va;  // same after normalization
        out[tid] = x + y;
    }
}

// ------------------------------------------------------------------
// Strength reduction: multiply by power of 2 → shift.
// v * 4 → v << 2, v * 8 → v << 3.

__global__ void strength_reduce(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int a = v * 4;    // should become v << 2
        int b = v * 8;    // should become v << 3
        out[tid] = a + b;
    }
}
