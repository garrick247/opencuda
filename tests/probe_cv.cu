// Probe: Patterns with local arrays and complex initialization
// - Local int array initialized with computed values
// - Local float array with partial initialization
// - Local array used as accumulator
// - Local array of pointers

__global__ void local_array_init(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Local array initialized element-by-element
        int table[8];
        for (int i = 0; i < 8; i++) {
            table[i] = (tid + i) * (i + 1);
        }
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += table[i];
        }
        out[tid] = sum;
    }
}

// 2D local array simulation (flat)
__global__ void local_matrix(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < 4) {
        float mat[4];
        for (int j = 0; j < 4; j++) {
            mat[j] = in[(tid * 4 + j) % n];
        }
        float row_sum = 0.0f;
        for (int j = 0; j < 4; j++) {
            row_sum += mat[j];
        }
        out[tid] = row_sum;
    }
}
