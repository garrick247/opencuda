// Probe: Patterns that test the expression parser with unusual groupings
// - ((((expr)))) — multiple levels of parenthesization
// - (a op b) op (c op d) — balanced binary tree of operations
// - Function call result immediately subscripted: func()[i] — can we handle this?
// - Nested ternary with parentheses: (a ? b : c) ? d : e

__device__ int pick(int *arr, int idx) {
    return arr[idx];
}

__global__ void expr_grouping(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid];
        
        // Multiple levels of parentheses
        int r1 = ((((va + vb))));
        
        // Balanced tree
        int r2 = (va + vb) * (vc - vb) + (va - vc) * (vb + vc);
        
        // Nested ternary with parens
        int cond1 = (va > vb) ? 1 : 0;
        int cond2 = (vb > vc) ? 1 : 0;
        int r3 = (cond1 ? va : vb) + (cond2 ? vb : vc);
        
        out[tid] = r1 + r2 + r3;
    }
}

// Chained comparisons via intermediate variables
__global__ void multi_cmp(int *out, int *a, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = a[tid];
        // Simulate: -10 <= v <= 10
        int in_range = (v >= -10) && (v <= 10);
        // Simulate: 0 <= v < n
        int valid = (v >= 0) && (v < n);
        out[tid] = in_range * 2 + valid;
    }
}
