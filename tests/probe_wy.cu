// Probe: multi-level pointer indexing, sizeof in expressions, string printf,
// device fn table dispatch (simulated via switch), and edge cases in
// integer type promotion.

// ------------------------------------------------------------------
// 2D array via flat pointer: arr[row][col] → arr[row * cols + col].

__global__ void flat_2d_rw(float *arr, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        arr[r * cols + c] = (float)(r * cols + c);
    }
}

// ------------------------------------------------------------------
// sizeof in arithmetic: sizeof(int) is 4, sizeof(float) is 4, sizeof(double) is 8.

__global__ void sizeof_arith(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = sizeof(int);       // 4
        int b = sizeof(float);     // 4
        int c = sizeof(double);    // 8
        int d = sizeof(long long); // 8
        out[tid] = a + b + c + d + tid;  // 24 + tid
    }
}

// ------------------------------------------------------------------
// Integer promotion edge cases: char + int, short * int.

__global__ void int_promotion(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char  sc = (signed char)(tid & 0x7F);
        unsigned char uc = (unsigned char)(tid & 0xFF);
        short         s  = (short)(tid * 2);
        unsigned short us = (unsigned short)(tid * 3);

        int r = sc + uc + (int)s + (int)us;
        out[tid] = r;  // sc + uc + 2*tid + 3*tid = 5*tid + (uc - sc may differ at boundaries)
    }
}

// ------------------------------------------------------------------
// Function dispatch via switch (simulating virtual dispatch).

__device__ int dispatch(int op, int a, int b) {
    switch (op & 7) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return (b != 0) ? a / b : 0;
        case 4: return a & b;
        case 5: return a | b;
        case 6: return a ^ b;
        case 7: return a > b ? a : b;
        default: return 0;
    }
}

__global__ void dispatch_kernel(int *out, int *ops, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dispatch(ops[tid], a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Unary NOT and double NOT.

__global__ void unary_not(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = !v;       // logical not: 0 → 1, nonzero → 0
        int b = !!v;      // double not: normalizes to 0 or 1
        int c = ~v;       // bitwise not
        out[tid] = a + b * 2 + (c & 0xFF) * 4;
    }
}

// ------------------------------------------------------------------
// Struct with static-sized array, passed by pointer.

struct Matrix2x2 {
    float data[4];  // row-major: [0][0],[0][1],[1][0],[1][1]
};

__device__ void mat2x2_mul(struct Matrix2x2 *C,
                            struct Matrix2x2 *A,
                            struct Matrix2x2 *B) {
    C->data[0] = A->data[0]*B->data[0] + A->data[1]*B->data[2];
    C->data[1] = A->data[0]*B->data[1] + A->data[1]*B->data[3];
    C->data[2] = A->data[2]*B->data[0] + A->data[3]*B->data[2];
    C->data[3] = A->data[2]*B->data[1] + A->data[3]*B->data[3];
}

__global__ void mat2x2_kernel(float *out, float *in_A, float *in_B, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Matrix2x2 A, B, C;
        int base = tid * 4;
        A.data[0] = in_A[base+0]; A.data[1] = in_A[base+1];
        A.data[2] = in_A[base+2]; A.data[3] = in_A[base+3];
        B.data[0] = in_B[base+0]; B.data[1] = in_B[base+1];
        B.data[2] = in_B[base+2]; B.data[3] = in_B[base+3];
        mat2x2_mul(&C, &A, &B);
        out[base+0] = C.data[0]; out[base+1] = C.data[1];
        out[base+2] = C.data[2]; out[base+3] = C.data[3];
    }
}

// ------------------------------------------------------------------
// Loop with multiple break conditions and a flag.

__global__ void multi_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int found = -1;
        for (int i = 0; i < 32; i++) {
            if (v == 0) { found = -2; break; }
            if (v == i) { found = i; break; }
            if (v < 0)  { found = -3; break; }
            v = (v * 17 + 3) & 0xFF;
        }
        out[tid] = found;
    }
}

// ------------------------------------------------------------------
// Ternary chain with pointer ops.

__global__ void ternary_ptr(float *out, float *a, float *b, float *c,
                             int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *src = (sel[tid] == 0) ? a :
                     (sel[tid] == 1) ? b : c;
        out[tid] = src[tid] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Accumulate into multiple output arrays with predication.

__global__ void multi_output_pred(int *pos_sum, int *neg_sum, int *zero_count,
                                   int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > 0) atomicAdd(pos_sum, v);
        else if (v < 0) atomicAdd(neg_sum, v);
        else atomicAdd(zero_count, 1);
    }
}
