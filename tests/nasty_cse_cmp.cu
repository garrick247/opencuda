// Tests CmpInst CSE: same comparison computed twice in compound boolean.
// In (x > 0 && x > 0), both comparisons are in the same block — should CSE.
__global__ void cmp_dedup(int *out, int *a, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int v = a[tid];
    // Compound condition with duplicate sub-expression: v > 0 appears twice.
    // Both CmpInsts land in the same block before the CondBrTerm — should CSE.
    if (v > 0 && v > 0) {
        out[tid] = v;
    } else {
        out[tid] = 0;
    }
}

__global__ void cmp_commutative(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int x = a[tid];
    int y = b[tid];
    // EQ is commutative: (x == y) and (y == x) should CSE.
    if (x == y && y == x) {
        out[tid] = 1;
    } else {
        out[tid] = 0;
    }
}
