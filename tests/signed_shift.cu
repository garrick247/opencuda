// Regression: signed right shift must use shr.s32 (arithmetic, sign-extending),
// not shr.b32 (logical, zero-filling).
// Without this fix: (-4 >> 1) would give 2147483646 instead of -2.
__global__ void signed_shift(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        out[tid] = val >> 2;   // arithmetic right shift — must use shr.s32
    }
}

// Unsigned right shift must stay shr.b32 (logical).
__global__ void unsigned_shift(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int val = in[tid];
        out[tid] = val >> 2;   // logical right shift — must use shr.b32
    }
}

// Regression: int x >> unsigned_amount must use shr.s32, not shr.b32.
// Without this fix: shifting an INT32 by a UINT32 amount (e.g. threadIdx.x & 0x1F)
// would produce shr.b32 (logical) because _result_type(INT32, UINT32) = UINT32.
__global__ void signed_shift_uint_amount(int *out, int *in, int n) {
    unsigned int tid = threadIdx.x;
    if (tid < (unsigned int)n) {
        int val = in[tid];
        unsigned int shift = tid & 0x1F;  // UINT32 shift amount
        out[tid] = val >> shift;           // must still use shr.s32 (left operand is INT32)
    }
}
