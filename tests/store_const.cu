// Regression: storing a constant value to global memory.
// PTX st instruction requires a register source — immediate operands are invalid.
// Without fix: st.global.s32 [%rd0], 42  (immediate rejected by ptxas)
__global__ void store_const_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = 42;       // store constant int
        out[tid + 1] = -1;   // store negative constant
        out[tid + 2] = 0;    // store zero
    }
}
