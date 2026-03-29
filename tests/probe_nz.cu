// Probe: 64-bit shift with variable amount, namespace/scope resolution,
// and SHL/SHR result type correctness (left operand type preserved).

// ------------------------------------------------------------------
// 64-bit value shifted by a variable amount (not a constant).
// PTX shl.b64/shr.b64 require the shift amount to be b32 — NOT b64.
// If the compiler widens the shift amount to b64, ptxas will reject it.

__global__ void shift64_var(unsigned long long *out, unsigned long long *data,
                             int *amounts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = data[tid];
        int shift = amounts[tid];          // b32 shift count
        out[tid*2+0] = v << shift;         // shl.b64 with b32 reg amount
        out[tid*2+1] = v >> shift;         // shr.b64 with b32 reg amount
    }
}

// ------------------------------------------------------------------
// Left-shift result type is left operand type (C §6.5.7).
// INT32 << UINT32 should still produce INT32 result (not UINT32).
// If the compiler uses _result_type(lhs, rhs) for shifts, it would
// incorrectly pick UINT32, causing shr.b32 instead of shr.s32.

__global__ void shift_result_type(int *out, int *data, unsigned int *amounts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        unsigned int shift = amounts[tid];
        // INT32 >> UINT32: result should be INT32 (arithmetic shift right)
        // Correct: shr.s32    Wrong: shr.b32 (logical shift, no sign extend)
        out[tid] = v >> shift;
    }
}

// ------------------------------------------------------------------
// Signed integer division and modulo with negative operands.
// C truncated semantics: -7 / 2 = -3 (not -4), -7 % 2 = -1 (not 1).
// Tests constant folding uses C-style truncation not Python floor.

__global__ void trunc_div_test(int *out) {
    // These are constant expressions that get folded at compile time
    out[0] = -7 / 2;    // C: -3
    out[1] = -7 % 2;    // C: -1
    out[2] = 7 / -2;    // C: -3
    out[3] = 7 % -2;    // C: 1
    out[4] = -7 / -2;   // C: 3
}
