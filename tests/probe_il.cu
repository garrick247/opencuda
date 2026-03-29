// Probe: compound assignments on variables bound to expression-result Values
// (catches the same class as the i++/j-- bug — _variables["x"] = computed_val
//  where computed_val.name != "x", then x += y should update _variables["x"])

__global__ void expr_compound(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // Both a and b are initialized by expressions (not literals)
        int a = n * 2;     // a = Value("compound")
        int b = n + 1;     // b = Value("compound")
        a += b;            // should update _variables["a"], not _variables["compound"]
        b *= 3;            // should update _variables["b"]
        out[0] = a;
        out[1] = b;
    }
}

__global__ void expr_compound2(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid] * 2.0f;  // x = Value("compound")
        float y = in[tid] + 1.0f;  // y = Value("compound")
        x += y;            // should update _variables["x"]
        x -= 0.5f;
        out[tid] = x * y;
    }
}

// Cascaded: result used in next computation
__global__ void cascaded_assign(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = n - 1;
        int b = a + 2;
        int c = b * 3;
        a += b;
        b += c;
        c += a;
        out[0] = a;
        out[1] = b;
        out[2] = c;
    }
}
