// Probe: CSE across complex expressions, loop-carried CSE safety,
// mixed signed/unsigned arithmetic, and type coercion in array index.

// ------------------------------------------------------------------
// CSE with multiple operands: same subexpression used in two different
// outer expressions.  The common part (tid*cols + j) should CSE.

__global__ void cse_index(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        int base = tid * cols;
        // Both uses of (base + 0) should not duplicate the multiply
        int a = data[base];
        int b = data[base + 1];
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// Loop-carried CSE safety: two variables with the same initial expression
// but different loop updates.  CSE must NOT merge their registers.

__global__ void cse_safe_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = 0;   // loop-carried
        int y = 0;   // loop-carried
        for (int i = 0; i < n; i++) {
            x += data[i];
            y += data[i] * 2;
        }
        out[0] = x;
        out[1] = y;
    }
}

// ------------------------------------------------------------------
// Signed/unsigned mix: signed int compared against unsigned constant.
// Checks that the comparison uses correct setp type.

__global__ void signed_unsigned_mix(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        unsigned int u = (unsigned int)v;
        // Signed comparison
        int s_positive = (v > 0) ? 1 : 0;
        // Unsigned comparison (same bit pattern, different semantics for negatives)
        int u_big = (u > 2147483648u) ? 1 : 0;  // > INT_MAX as unsigned
        out[tid * 2 + 0] = s_positive;
        out[tid * 2 + 1] = u_big;
    }
}

// ------------------------------------------------------------------
// Array index from long computation: uses a 64-bit index.
// Tests that the index computation correctly stays in 64-bit.

__global__ void wide_index(long long *out, long long *data, long long stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long idx = (long long)tid * stride;
        out[tid] = data[idx];
    }
}

// ------------------------------------------------------------------
// Modulo in array index: circular buffer access.
// Tests that mod is handled correctly and the address is valid.

__global__ void circ_read(int *out, int *buf, int buf_size, int *offsets, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = offsets[tid] % buf_size;
        out[tid] = buf[idx];
    }
}
