// Probe: Unusual preprocessor + lexer edge cases
// - Number with exponent but no decimal: 1e5f
// - Hex float literals: 0x1p10f (hexadecimal float) — should fail gracefully
// - Very small float: 1e-38f
// - Multi-digit hex: 0xDEADBEEF
// - Octal literal: 0755

__global__ void numeric_edge_cases(float *fout, int *iout, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = 1e5f;
        float b = 1.0e-38f;
        float c = 3.14e2f;
        int d = 0xDEADBEEF;  // 0xDEADBEEF = -559038737 as int32
        int e = 0755;  // octal 755 = 493 decimal
        fout[tid] = a + b + c + (float)tid;
        iout[tid] = d ^ e ^ tid;
    }
}

// Float arithmetic that could trigger precision edge cases
__global__ void float_precision(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = (float)tid;
        // Large multiplication then small addition (float precision stress)
        float r = x * 1e6f + 1e-6f;
        out[tid] = r;
    }
}
