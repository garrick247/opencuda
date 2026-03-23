// Liveness test: x defined before if, live across both branches, used after merge.
// y defined in one branch, must not alias x's physical register since x is still live.
__global__ void branch_overlap(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid];        // live across branches
        float y;
        if (x > 0.0f) {
            y = x * 2.0f;         // x still live here
        } else {
            y = -x;               // x still live here
        }
        out[tid] = x + y;         // x and y both live at merge
    }
}
