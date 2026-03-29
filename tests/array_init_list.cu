// Regression: local array aggregate initializer — int arr[4] = {1, 2, 3, 4};
// Without fix: parser expected SEMI after array size brackets, got ASSIGN '=' →
//   ParseError "expected SEMI, got ASSIGN '='".
// Fix: local array declaration code checks for '= {' after array dims; if found,
//   parses { expr, ... } and emits StoreInst for each element into .local memory.

__global__ void array_init_int(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int weights[4] = {1, 2, 4, 8};
        int sum = 0;
        int i;
        for (i = 0; i < 4; i++) sum += weights[i];
        out[tid] = sum;  // should be 15
    }
}

__global__ void array_init_float(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float coeff[3] = {0.25f, 0.5f, 0.25f};
        float result = coeff[0] + coeff[1] + coeff[2];
        out[tid] = result;  // should be 1.0
    }
}

__global__ void array_init_partial(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Partial initializer — remaining elements unspecified (but no crash)
        float v[4] = {1.0f, 2.0f};
        float s = v[0] + v[1];
        out[tid] = s;
    }
}

__global__ void array_init_dynamic(float *data, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float buf[4] = {data[tid*4+0], data[tid*4+1], data[tid*4+2], data[tid*4+3]};
        float sum = buf[0] + buf[1] + buf[2] + buf[3];
        out[tid] = sum;
    }
}
