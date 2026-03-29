// Regression: __constant__ __device__ reversed qualifier order + ld.const state space
// Without fix 1: ParseError "expected type, got '__device__'"
//   — _parse_constant_decl didn't accept __device__ as qualifier after __constant__
// Without fix 2: ptxas "State space mismatch" on ld instruction
//   — LoadInst via SymbolRef was always emitted as ld.global even for .const symbols
// Fix 1: _parse_constant_decl skips KW_DEVICE qualifiers
// Fix 2: emit.py LoadInst checks SymbolRef.ty.addr_space and emits ld.const

__device__ __constant__ int dc_int = 100;
__constant__ __device__ float cd_float = 2.5f;
__constant__ float c_only = 1.5f;

__global__ void const_qual_test(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = (float)dc_int * cd_float + c_only * (float)tid;
        out[tid] = v;
    }
}

// Reversed __constant__ qualifier order in array declaration
__constant__ __device__ int cd_table[8] = {1, 2, 3, 4, 5, 6, 7, 8};

__global__ void const_table_test(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 7;
        out[tid] = cd_table[i] * tid;
    }
}
