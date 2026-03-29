// Probe: static local variables, complex nested loops, device fn with
// multiple pointer outputs, __launch_bounds__ stacking, large local arrays,
// and sign extension / zero extension patterns.

// ------------------------------------------------------------------
// Static local variable: persists across calls (maps to .global in PTX).

__device__ int g_call_count = 0;

__global__ void static_counter(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        g_call_count++;
        out[0] = g_call_count;
    }
}

// ------------------------------------------------------------------
// Local array with non-constant but small range indexing.

__global__ void local_array_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[8];
        // Fill with data
        for (int i = 0; i < 8; i++) {
            buf[i] = data[tid] + i;
        }
        // Accumulate
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += buf[i];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Nested loops: matrix multiply 4x4 with inner accumulation.

__global__ void matmul4(float *C, float *A, float *B) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < 4 && col < 4) {
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            sum += A[row * 4 + k] * B[k * 4 + col];
        }
        C[row * 4 + col] = sum;
    }
}

// ------------------------------------------------------------------
// Device function with multiple outputs via pointers.

__device__ void divmod(int num, int denom, int *quot, int *rem) {
    *quot = num / denom;
    *rem  = num % denom;
}

__global__ void divmod_kernel(int *qout, int *rout, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int q, r;
        divmod(data[tid], 7, &q, &r);
        qout[tid] = q;
        rout[tid] = r;
    }
}

// ------------------------------------------------------------------
// Sign extension: char/short loads and sign-extend to int.

__global__ void sign_ext(int *out, signed char *in8, short *in16, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v8  = (int)in8[tid];   // sign-extend s8 → s32
        int v16 = (int)in16[tid];  // sign-extend s16 → s32
        out[tid] = v8 + v16;
    }
}

// ------------------------------------------------------------------
// Zero extension: unsigned char/short.

__global__ void zero_ext(unsigned int *out, unsigned char *in8,
                          unsigned short *in16, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v8  = (unsigned int)in8[tid];
        unsigned int v16 = (unsigned int)in16[tid];
        out[tid] = v8 * v16;
    }
}

// ------------------------------------------------------------------
// Byte stores: write individual bytes.

__global__ void byte_store(signed char *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (signed char)(data[tid] & 0xFF);
    }
}

// ------------------------------------------------------------------
// Short stores: write 16-bit values.

__global__ void short_store(short *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (short)(data[tid] & 0xFFFF);
    }
}

// ------------------------------------------------------------------
// Nested break/continue with labels emulated by flags.

__global__ void nested_search(int *found, int *data, int target, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows) {
        int base = tid * cols;
        int f = -1;
        for (int j = 0; j < cols; j++) {
            if (data[base + j] == target) {
                f = base + j;
                break;
            }
        }
        found[tid] = f;
    }
}
