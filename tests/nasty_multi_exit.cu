// Nasty: loop with three independent break conditions.
// Tests that multiple break paths all target the same correct exit block
// and that register state at each break is consistent.
__global__ void find_range(int* data, int* out_lo, int* out_hi, int n,
                            int lo_target, int hi_target) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int lo_found = -1;
    int hi_found = -1;
    int count = 0;

    for (int i = 0; i < n; i++) {
        int v = data[i];
        if (v < lo_target) {
            lo_found = i;
            break;              // exit 1: underflow
        }
        if (v > hi_target) {
            hi_found = i;
            break;              // exit 2: overflow
        }
        count = count + 1;
        if (count > n / 2) {
            break;              // exit 3: iteration budget
        }
    }
    out_lo[tid] = lo_found;
    out_hi[tid] = hi_found;
}
