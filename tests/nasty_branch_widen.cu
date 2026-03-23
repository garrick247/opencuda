// Nasty: int value computed pre-branch, used with pointer arithmetic in both
// arms of a conditional, then the int is used again after the merge.
// Tests that the widened pointer register is consistently declared and that
// the pre-branch int Value is still live at the post-merge store.
__global__ void split_store(float* pos_out, float* neg_out, float* in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float v = in[tid];
    int idx = tid * 4;          // int that must stay live across the branch
    if (v >= 0.0f) {
        float* p = pos_out;
        int offset = idx;       // reuses pre-branch int in branch arm
        p[tid] = v;
    } else {
        float* q = neg_out;
        int offset = idx;       // same pre-branch int, other arm
        q[tid] = v * -1.0f;
    }
    // idx still live here (not eliminated by CSE across branches)
    int sentinel = idx + 1;
    pos_out[n] = sentinel;      // use idx after merge
}
