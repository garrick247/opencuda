// Switch inside a for-loop: tests switch writeback + loop back-edge liveness.
__global__ void switch_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int acc = 0;
    for (int i = 0; i < n; i++) {
        int v = data[i];
        int delta;
        switch (v % 3) {
            case 0: delta = 1; break;
            case 1: delta = 2; break;
            default: delta = 3; break;
        }
        acc = acc + delta;
    }
    out[tid] = acc;
}
