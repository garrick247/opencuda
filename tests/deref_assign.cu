// Regression: *ptr = value — dereference as lvalue (write through pointer)
// Without fix: _parse_lvalue_or_expr saw STAR IDENT but didn't handle it;
//   _parse_primary_expr received '=' token → ParseError "unexpected token '='".
// Fix: _parse_lvalue_or_expr checks for STAR followed by a PtrTy IDENT;
//   returns the pointer directly so _parse_assign_expr can emit StoreInst.

// Simple *ptr assignment from pointer parameter
__global__ void deref_write(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *dst = out + tid;
        int *src = in + tid;
        *dst = *src * 2;
    }
}

// Float pointer deref write
__global__ void deref_float(float *out, float *in, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = out + tid;
        *p = in[tid] * scale;
    }
}

// Device function writing via dereferenced output pointer
__device__ void compute_pair(int *a, int *b, int x, int y) {
    *a = x + y;
    *b = x * y;
}

__global__ void use_compute_pair(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b;
        compute_pair(&out[tid*2], &out[tid*2+1], in[tid], tid);
    }
}
