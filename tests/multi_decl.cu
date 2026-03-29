// Regression: multiple variable declarations on one line (int a=1, b=2;)
// Without fix: parser fails with "expected SEMI, got COMMA" after first declarator.
__global__ void multi_decl_test(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 1, b = 2, c = 3;
        float x = 1.0f, y = 2.0f;
        int sum = a + b + c;
        out[tid * 2 + 0] = x + y + (float)sum;
        out[tid * 2 + 1] = (float)(a * b);
    }
}
