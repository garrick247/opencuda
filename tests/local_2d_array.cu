// Regression: local 2D array declaration and indexing: float buf[4][4]
// Without fix: float buf[4][4] → ParseError "expected SEMI, got LBRACKET '['"
//   because the local array declaration only consumed one [N].
// Fix: local array parser now handles [d0][d1]... multi-dim syntax:
//   - Flat size = d0 * d1 * ... for .local allocation
//   - Row stride = product(d1..) * elem_size for arr[i][j] indexing

__global__ void local_2d(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float buf[4][4];
        int i, j;
        // Fill from global memory
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                buf[i][j] = in[tid * 16 + i * 4 + j];
            }
        }
        // Sum all elements
        float sum = 0.0f;
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                sum += buf[i][j];
            }
        }
        out[tid] = sum;
    }
}

__global__ void prefix_sum_2d(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int scratch[8][2];  // int 2D array
        int i;
        for (i = 0; i < 8; i++) {
            scratch[i][0] = in[tid * 8 + i];
            scratch[i][1] = 0;
        }
        // Running prefix
        int s = 0;
        for (i = 0; i < 8; i++) {
            s += scratch[i][0];
            scratch[i][1] = s;
        }
        out[tid] = scratch[7][1];
    }
}
