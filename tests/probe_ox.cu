// Probe: ternary in loop carried position, multi-dim shared mem initializer,
// unsigned comparison edge cases, and __device__ fn with pointer output param.

// ------------------------------------------------------------------
// Ternary used as loop-carried value update.
// sum = (v > 0) ? sum + v : sum  — conditional accumulation via ternary.

__global__ void ternary_carry(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i];
            sum = (v > 0) ? sum + v : sum;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Unsigned comparisons: u32 and u64 boundary values.
// Tests setp.lo/hi/ls/hs for unsigned relations.

__global__ void unsigned_cmp(unsigned int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = data[tid];
        unsigned int result = 0;
        if (v < 0x80000000u) result |= 1;   // lo  (< half of u32 range)
        if (v > 0x80000000u) result |= 2;   // hi
        if (v <= 0xFFFF0000u) result |= 4;  // ls
        if (v >= 0x00010000u) result |= 8;  // hs
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// __device__ function that writes result through a pointer parameter.
// Tests that pointer-typed parameters are passed correctly to inlines.

__device__ void minmax(int *lo, int *hi, int a, int b) {
    *lo = (a < b) ? a : b;
    *hi = (a < b) ? b : a;
}

__global__ void ptr_out_param(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n - 1) {
        int lo, hi;
        minmax(&lo, &hi, data[tid], data[tid + 1]);
        out[tid * 2 + 0] = lo;
        out[tid * 2 + 1] = hi;
    }
}

// ------------------------------------------------------------------
// Loop over fixed range using unsigned induction variable.
// Tests that `unsigned int i` loop behaves correctly (no sign extension).

__global__ void unsigned_loop(float *out, float *data, unsigned int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned int i = 0u; i < n; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Bit-field extraction pattern using shifts and masks.
// Common in GPU code: extract R/G/B from packed RGBA.

__global__ void rgba_unpack(unsigned int *out, unsigned int *packed, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int p = packed[tid];
        unsigned int r = (p >> 24) & 0xFF;
        unsigned int g = (p >> 16) & 0xFF;
        unsigned int b = (p >>  8) & 0xFF;
        unsigned int a =  p        & 0xFF;
        out[tid * 4 + 0] = r;
        out[tid * 4 + 1] = g;
        out[tid * 4 + 2] = b;
        out[tid * 4 + 3] = a;
    }
}
