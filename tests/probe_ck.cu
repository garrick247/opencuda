// Probe: Patterns that expose issues with optimizer passes
// - LICM: loop-invariant computation should be hoisted
// - CSE: common subexpressions within a block
// - Constant folding: arithmetic on constants
// - Dead code: code after unconditional return

__global__ void licm_test(float *out, float *in, float *params, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // params[0] and params[1] are loop-invariant
        float scale = params[0] * params[1];  // should be hoisted
        float offset = params[0] + params[1];  // should be hoisted
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            sum += in[(tid + i) % n] * scale + offset;
        }
        out[tid] = sum;
    }
}

__global__ void cse_test(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid];
        // Common subexpression: va + vb computed 3 times
        int sum1 = va + vb;
        int sum2 = va + vb;  // same as sum1
        int sum3 = va + vb;  // same as sum1
        out[tid] = sum1 * sum2 + sum3;
    }
}

__global__ void const_fold_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // These should all fold to constants
        int a = 10 * 10;      // 100
        int b = 256 / 4;      // 64
        int c = 1000 - 500;   // 500
        int d = a + b + c;    // 664
        out[tid] = d + tid;
    }
}
