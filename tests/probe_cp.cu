// Probe: Unusual expressions and operator combinations
// - Bitwise NOT of variable: ~x
// - Compound expression: a & b | c ^ d
// - Left-shift by constant: x << 2
// - Right-shift of negative value: n >> 1 (arithmetic shift)
// - Nested ternary in array subscript
// - Conditional in left-hand side of assignment? (no, but as RHS)

__global__ void bit_ops_chain(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid];
        int not_a = ~va;
        int or_bc = vb | vc;
        int and_not = not_a & or_bc;
        int xor_all = va ^ vb ^ vc;
        int shift_left = va << 2;
        int shift_right = va >> 1;
        out[tid] = and_not + xor_all + shift_left + shift_right;
    }
}

__global__ void conditional_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Index computed by ternary
        int idx = (tid < n / 2) ? tid * 2 : (n - 1 - tid) * 2 + 1;
        out[tid] = in[idx % n];
    }
}

// Complex boolean expression
__global__ void complex_bool(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid];
        int result = (va > 0 && vb > 0) ? 1 :
                     (va < 0 || vb < 0) ? -1 :
                     (vc != 0) ? 2 : 0;
        out[tid] = result;
    }
}
