// Probe: 2D local arrays (int mat[4][4]), pointer-to-pointer,
// conditional struct initialization, and complex address arithmetic.

// ------------------------------------------------------------------
// 2D local array: int mat[4][4], accessed as mat[i][j].

__global__ void mat4x4_local(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int mat[4][4];
        // Fill with simple pattern
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat[i][j] = v + i * 4 + j;
            }
        }
        // Trace diagonal
        int diag = 0;
        for (int i = 0; i < 4; i++) {
            diag += mat[i][i];
        }
        out[tid] = diag;  // = (v+0) + (v+5) + (v+10) + (v+15) = 4v+30
    }
}

// ------------------------------------------------------------------
// 1D local array used with indirect index.

__global__ void indirect_local_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lut[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        int v = in[tid];
        int idx = v & 7;
        out[tid] = lut[idx];
    }
}

// ------------------------------------------------------------------
// Pointer-to-pointer: int **pp, read *pp then **pp.

__global__ void double_ptr_read(int *out, int **pp_array, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = pp_array[tid];   // first dereference
        int v = *p;               // second dereference
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Conditional struct initialization.

struct Point { int x; int y; };

__device__ struct Point make_point(int x, int y) {
    struct Point p;
    p.x = x;
    p.y = y;
    return p;
}

__global__ void cond_struct_init(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        struct Point p;
        if (v > 0) {
            p = make_point(v, v + 1);
        } else {
            p = make_point(-v, -v + 1);
        }
        out[tid] = p.x + p.y;
    }
}

// ------------------------------------------------------------------
// Array of pointers: float *arr[4], each points into a global array.

__global__ void array_of_ptrs(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Simulate array of pointers with computed offsets
        float *p0 = data + tid * 4 + 0;
        float *p1 = data + tid * 4 + 1;
        float *p2 = data + tid * 4 + 2;
        float *p3 = data + tid * 4 + 3;
        out[tid] = *p0 + *p1 + *p2 + *p3;
    }
}

// ------------------------------------------------------------------
// Local float matrix multiply (2x2).

__global__ void mat2x2_local(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a[2][2], b[2][2], c[2][2];
        // Init a and b from input
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                a[i][j] = in[tid * 4 + i * 2 + j];
                b[i][j] = in[tid * 4 + i * 2 + j] + 1.0f;
            }
        }
        // C = A * B
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                c[i][j] = 0.0f;
                for (int k = 0; k < 2; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        // Sum all elements of C
        float sum = c[0][0] + c[0][1] + c[1][0] + c[1][1];
        out[tid] = sum;
    }
}
