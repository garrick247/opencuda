// Probe: operator precedence edge cases, conditional expression nesting,
// and compound assignment with non-trivial RHS.

// ------------------------------------------------------------------
// Precedence: & vs == (& binds tighter).

__global__ void prec_bitand_eq(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // (v & 3) == 0 : check low 2 bits zero
        int r0 = (v & 3) == 0;
        // v & (3 == 0) = v & 0 = 0 — different grouping via parens
        int r1 = v & (3 == 0);
        // v & 1 == 0 : without parens, == has higher prec than &?
        // No! In C: & < == — so this is v & (1 == 0) = v & 0 = 0
        // but with explicit parens: (v & 1) == 0
        int r2 = (v & 1) == 0;
        out[tid] = r0 + r1 * 2 + r2 * 4;
    }
}

// ------------------------------------------------------------------
// Precedence: | vs ternary.

__global__ void prec_bitor_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // a | b ? c : d — ternary has lower prec than |
        // so this is (a | b) ? c : d
        int r = (v | 0xFF) ? 1 : 0;
        // Explicit nesting
        int s = v | (v > 0 ? 0x100 : 0x200);
        out[tid] = r + s;
    }
}

// ------------------------------------------------------------------
// Compound assignment with complex RHS.

__global__ void compound_complex(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        x += y * y + 1;
        y -= x >> 2;
        x ^= y & 0xFF;
        y |= (x > 0) ? 0x1000 : 0;
        out[tid * 2 + 0] = x;
        out[tid * 2 + 1] = y;
    }
}

// ------------------------------------------------------------------
// Nested ternary (right-associative).

__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Equivalent to if-else-if chain
        int r = v < 0 ? -1 : v == 0 ? 0 : v < 10 ? 1 : v < 100 ? 2 : 3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Comma in for-loop: multiple init, multiple increment.

__global__ void for_multi_init(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int i, j;
        for (i = 0, j = v; i < 8 && j > 0; i++, j--) {
            acc += i * j;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Post-increment in complex expression.

__global__ void postinc_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0;
        int a = v + i++;      // a = v+0, i=1
        int b = v + ++i;      // i=2, b = v+2
        int c = (i++) * v;    // c = 2*v, i=3
        out[tid] = a + b + c + i;  // v + (v+2) + 2v + 3
    }
}

// ------------------------------------------------------------------
// Assignment in condition (C idiom: while ((c = next()) != 0)).

__global__ void assign_in_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0, x;
        int i = 0;
        // Use assignment in while condition
        while ((x = v - i * 3) > 0 && i < 10) {
            sum += x;
            i++;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Array subscript with complex index expression.

__global__ void complex_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = tid;
        int v = in[i];
        // Multi-level index math
        int a = in[(i + 1) % n];
        int b = in[(i * 2 + 1) % n];
        int c = in[n - 1 - i];
        out[i] = v + a + b + c;
    }
}
