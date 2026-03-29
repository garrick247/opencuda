// Probe: edge cases that are more likely to produce wrong PTX than parse errors
// - Integer overflow arithmetic (wrap behavior)
// - Shift by variable amount
// - Float-to-int with rounding modes
// - Comparison of different widths (int vs long long)
// - Mixing signed and unsigned in comparisons

__global__ void shift_by_var(unsigned int *out, unsigned int *in, int *shifts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        int sh = shifts[tid] & 31;  // safe shift amount
        out[tid] = (v << sh) | (v >> (32 - sh));  // rotate
    }
}

__global__ void mixed_width_cmp(long long *out, int *a, long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long va = (long long)a[tid];
        long long vb = b[tid];
        out[tid] = (va < vb) ? va : vb;
    }
}

__global__ void unsigned_signed_mix(int *out, unsigned int *u, int *s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int uv = u[tid];
        int sv = s[tid];
        // Mix: cast unsigned to signed for comparison
        int uv_s = (int)uv;
        out[tid] = (uv_s > sv) ? uv_s - sv : sv - uv_s;
    }
}
