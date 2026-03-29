// Probe: many live variables (register pressure), complex expression trees,
// deeply nested ternary, loop unroll with complex body,
// expression with all arithmetic operators

// Many live variables at once
__global__ void many_vars(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = a + 1;
        int c = b * 2;
        int d = c - 3;
        int e = d / 4;
        int f = e + b;
        int g = c + d;
        int h = g - f;
        int i2 = h * e;
        int j = i2 + a;
        int k = j - b;
        int l = k * 2;
        int m = l + d;
        // Use all 13 variables
        out[tid] = a + b + c + d + e + f + g + h + i2 + j + k + l + m;
    }
}

// Deep expression tree: all binary ops combined
__global__ void all_ops(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int add_r = v + 10;
        int sub_r = v - 5;
        int mul_r = v * 3;
        int div_r = (v != 0) ? (v / 2) : 0;
        int mod_r = (v != 0) ? (v % 7) : 0;
        int and_r = v & 0xFF;
        int or_r  = v | 0x100;
        int xor_r = v ^ 0xAA;
        int shl_r = v << 2;
        int shr_r = v >> 1;
        out[tid] = add_r + sub_r + mul_r + div_r + mod_r +
                   and_r + or_r + xor_r + shl_r + shr_r;
    }
}

// Deeply nested ternary (chained)
__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Equivalent to: if v<0 -1; elif v<10 0; elif v<100 1; else 2
        int result = (v < 0) ? -1 :
                     (v < 10) ? 0 :
                     (v < 100) ? 1 : 2;
        out[tid] = result;
    }
}

// Unroll with complex body: multiple ops per iteration
__global__ void complex_unroll(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 0, c = 1;
        for (int i = 0; i < 8; i++) {
            a += i * i;           // sum of squares
            b += i * (i + 1);     // sum of i*(i+1)
            c *= (i % 3 == 0) ? 1 : (i % 3 == 1 ? 2 : 3);
        }
        out[0] = a;   // 0+1+4+9+16+25+36+49 = 140
        out[1] = b;   // 0+2+6+12+20+30+42+56 = 168
        out[2] = c;   // 1*1*2*3*1*2*3*1 = 36
    }
}
