// Probe: Variable shadowing in nested scopes,
// variable declared inside if/for body,
// re-declaration of outer variable in inner scope

__global__ void shadow_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int sum = 0;  // outer sum
    if (tid < n) {
        int x = in[tid];  // declared inside if
        for (int i = 0; i < x; i++) {
            int y = i * 2;  // declared inside for
            sum += y;
        }
    }
    if (tid < n) out[tid] = sum;
}

// Variable re-used across multiple if blocks
__global__ void reuse_var(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float val;
    if (tid < n) {
        val = in[tid];
    } else {
        val = 0.0f;
    }
    if (tid < n) {
        out[tid] = val * val;
    }
}

// Variables declared in for init, used after loop
__global__ void for_init_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int total = 0;
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            total += in[i] + in[j];
        }
        out[tid] = total;
    }
}

// Nested loops with same loop variable name
__global__ void nested_same_var(int *out, int *mat, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        int sum = 0;
        for (int i = 0; i < cols; i++) {
            sum += mat[tid * cols + i];
        }
        out[tid] = sum;
    }
}
