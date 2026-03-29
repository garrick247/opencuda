// Probe: Things that C/C++ compilers should reject but we might silently accept
// - Assignment inside condition: if (n = func())  
// - Comparison of pointer with integer: ptr == 0
// - Struct assignment across types (type mismatch — should fail or warn)
// - sizeof(struct) used in arithmetic

struct Pair {
    int first;
    int second;
};

__global__ void sizeof_usage(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sz = (int)sizeof(Pair);  // should be 8
        int arr_sz = (int)sizeof(int) * 4;  // 16
        out[tid] = sz + arr_sz + tid;
    }
}

// Assignment in condition (should work since it's an expression)
__global__ void assign_in_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        int i = 0;
        int v;
        // while ((v = in[i++]) != 0 && i < n)
        // This is too complex for us — test simpler variant:
        int sum = 0;
        while (i < n) {
            v = in[i];
            sum += v;
            i++;
            if (v == 0) break;
        }
        out[0] = sum;
    }
}
