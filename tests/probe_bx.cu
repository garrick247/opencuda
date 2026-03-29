// Probe: Edge cases in type conversion and promotion
// - int8 / uint8 arithmetic (treated as int32)
// - Explicit conversion chain: float -> int -> long long
// - Mixed signed/unsigned comparison (UB in C, but should not crash)
// - Float literal without explicit f suffix (double literal)
// - Float/double mixed arithmetic

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef short int16_t;

__global__ void byte_ops(int *out, uint8_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint8_t a = in[tid];
        uint8_t b = in[(tid + 1) % n];
        // Byte arithmetic (widened to int32)
        int sum = (int)a + (int)b;
        int diff = (int)a - (int)b;
        int prod = (int)a * (int)b;
        out[tid] = sum + diff + prod;
    }
}

__global__ void double_float_mix(double *dout, float *fout, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fv = fin[tid];
        double dv = (double)fv;
        double result = dv * 3.14159265358979;  // double literal
        fout[tid] = (float)result;
        dout[tid] = result;
    }
}
