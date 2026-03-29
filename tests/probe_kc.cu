// Probe: bitwise NOT (~x), one-sided if (some vars modified, others not),
// ternary as array subscript, modulo sign behavior,
// variable assigned in multiple sequential ifs without else

// Bitwise operations including NOT and XOR chain
__global__ void bitwise_chain(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int a = ~v;           // bitwise NOT
        unsigned int b = v ^ a;        // v XOR ~v = all ones
        unsigned int c = b & 0xFFFF;   // keep low 16 bits
        unsigned int d = ~c;           // NOT again
        out[tid] = d;
    }
}

// One-sided if: x modified in if-true only, y never modified
__global__ void one_sided_multi_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = 10, y = 20, z = 30;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) {
                x += v;    // x modified
                z += 1;    // z modified
                // y not touched
            }
            // after if: x and z may or may not have changed, y always 20
        }
        out[0] = x;
        out[1] = y;   // must still be 20
        out[2] = z;
    }
}

// Ternary as array subscript
__global__ void ternary_index(int *out, int *a, int *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sel[tid] == 0 → use a[tid], else → use b[tid]
        int idx = sel[tid] ? tid : tid;    // same index but through ternary
        int va = a[idx];
        int vb = b[idx];
        out[tid] = sel[tid] ? va : vb;
    }
}

// Modulo sign: in C, result has sign of dividend
__global__ void modulo_sign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // v % 3:  7%3=1,  -7%3=-1,  6%3=0,  -6%3=0
        out[tid] = v % 3;
    }
}

// Variable updated by multiple sequential one-sided ifs
__global__ void sequential_one_sided(int *out, int v) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int result = 0;
        if (v > 0)    result += 1;
        if (v > 10)   result += 10;
        if (v > 100)  result += 100;
        if (v > 1000) result += 1000;
        *out = result;
    }
}
