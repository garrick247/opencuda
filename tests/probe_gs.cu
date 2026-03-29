// Probe: predicate chains — sequential &&/|| building into a single condition,
// mixed int/bool in conditions, double negation

__global__ void pred_chain(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid], dv = d[tid];
        // Long predicate chain
        int r = (av > 0 && bv > 0 && cv > 0 && dv > 0) ? 1 : 0;
        int s = (av > 0 || bv > 0 || cv > 0 || dv > 0) ? 2 : 0;
        int t = (av > 0 && (bv > 0 || cv > 0) && dv > 0) ? 4 : 0;
        // Double negation
        int u = (!!(av) && !!(bv)) ? 8 : 0;
        out[tid] = r + s + t + u;
    }
}

// Ternary used as condition
__global__ void ternary_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Ternary result used as condition
        int cond = (v > 0) ? (v < 100 ? 1 : 2) : (v > -100 ? 3 : 4);
        if (cond & 1) {
            out[tid] = v * 2;
        } else {
            out[tid] = v + cond;
        }
    }
}

// Short-circuit evaluation (&&, || must not evaluate RHS if LHS decides)
__global__ void short_circuit(int *out, int *idx, int *arr, int n, int m) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        // Guard array access with bounds check
        int val = (i >= 0 && i < m) ? arr[i] : -1;
        out[tid] = val;
    }
}
