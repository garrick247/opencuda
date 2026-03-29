// Test: same integer index used for pointer arithmetic in two independent branches.
// If the widen_cache reuses a widened register across branches, one branch
// might use a register defined only in the other branch (undefined behavior).
__global__ void widen_crossblock_test(float *a, float *b, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float result;
        if (tid % 2 == 0) {
            // Branch A: uses tid as array index → should widen tid for ptr arith
            result = a[tid];
        } else {
            // Branch B: also uses tid as array index → needs its own widening
            result = b[tid];
        }
        out[tid] = result;
    }
}
