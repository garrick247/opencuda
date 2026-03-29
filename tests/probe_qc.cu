// Probe: complex expressions — deeply nested ternaries, compound
// assignments in unusual positions, predicate chains, bitfield-like
// packing patterns, and mixed expression forms.

// ------------------------------------------------------------------
// Deeply nested ternary: min of 4 values.

__global__ void min4_ternary(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid], vd = d[tid];
        int m = (va < vb ? va : vb) < (vc < vd ? vc : vd)
                ? (va < vb ? va : vb) : (vc < vd ? vc : vd);
        out[tid] = m;
    }
}

// ------------------------------------------------------------------
// Boolean expression tree: (a&&b) || (c&&d) || (e&&f).

__global__ void bool_tree(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int a = v & 1, b = v & 2, c = v & 4;
        int d = v & 8, e = v & 16, f = v & 32;
        int r = ((a && b) || (c && d) || (e && f)) ? 1 : 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Compound assignment in loop condition via &&.
// `while ((v = data[i]) != -1 && i < n)` — i and v both updated.

__global__ void cond_assign_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        int v = 0;
        while (i < n && (v = data[i]) != -1) {
            sum += v;
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Pre-increment and post-increment in expressions.

__global__ void pre_post_incr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int a = ++v;      // pre-increment: v becomes v+1, a = v+1
        int b = v++;      // post-increment: b = v, then v becomes v+1
        out[tid] = a + b; // a = (orig+1), b = (orig+1), result = 2*orig + 2
    }
}

// ------------------------------------------------------------------
// Bit packing: pack two 16-bit values into one 32-bit int.
// Tests that bit operations on constants fold correctly.

__global__ void bit_pack(int *out, int *lo, int *hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int l = lo[tid] & 0xFFFF;
        int h = hi[tid] & 0xFFFF;
        int packed = (h << 16) | l;
        out[tid] = packed;
    }
}

// ------------------------------------------------------------------
// Bit unpacking: extract high/low halves.

__global__ void bit_unpack(int *out_lo, int *out_hi, int *packed, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int p = packed[tid];
        out_lo[tid] = p & 0xFFFF;
        out_hi[tid] = (p >> 16) & 0xFFFF;
    }
}

// ------------------------------------------------------------------
// Saturating add: clamp sum to [0, 255].
// Tests that multi-ternary chains produce correct predicate sequences.

__global__ void sat_add(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = a[tid] + b[tid];
        int r = (s < 0) ? 0 : (s > 255) ? 255 : s;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Predicate chains: (a > 0) + (b > 0) + (c > 0) — count positives.
// Each comparison produces 0/1 via ternary.

__global__ void count_positive(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int cnt = (a[tid] > 0 ? 1 : 0)
                + (b[tid] > 0 ? 1 : 0)
                + (c[tid] > 0 ? 1 : 0);
        out[tid] = cnt;
    }
}

// ------------------------------------------------------------------
// Comma expression in for initializer: int i=0, j=n-1.
// Tests the multi-variable for-init pattern.

__global__ void two_ptr_walk(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0, j = n - 1; i <= j; i++, j--) {
            if (i == j) sum += data[i];
            else        sum += data[i] + data[j];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Ternary inside array index: data[a > b ? a : b].

__global__ void ternary_index(int *out, int *data, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = (a[tid] > b[tid]) ? a[tid] : b[tid];
        if (idx < n) {
            out[tid] = data[idx];
        } else {
            out[tid] = -1;
        }
    }
}
