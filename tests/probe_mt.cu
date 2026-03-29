// Probe: prefix ++/-- on struct fields
// - ++s.count in loop body
// - --s.count in conditional
// - prefix vs postfix equivalence for struct fields

struct Ctr { int pos, neg, zero; };

__global__ void prefix_inc_struct(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Ctr c; c.pos = 0; c.neg = 0; c.zero = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) ++c.pos;
            else if (v < 0) ++c.neg;
            else ++c.zero;
        }
        out[0] = c.pos; out[1] = c.neg; out[2] = c.zero;
    }
}
