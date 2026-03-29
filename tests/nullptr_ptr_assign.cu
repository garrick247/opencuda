// Regression: nullptr keyword and pointer variable reassignment (p = expr)
// Without fix 1: 'nullptr' not in _global_consts → ParseError "undefined variable 'nullptr'".
// Without fix 2: _parse_lvalue_or_expr consumed the pointer IDENT then fell through
//   to _parse_expr() without returning var → ParseError "unexpected token '='" for p = expr.
// Fix 1: added nullptr → Const(UINT64, 0) to Parser __init__.
// Fix 2: added 'return var' in the plain-pointer branch of _parse_lvalue_or_expr.

__global__ void nullptr_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int *p = nullptr;           // declare pointer initialized to nullptr
    if (tid < n) {
        if (in != nullptr) {    // compare pointer against nullptr
            p = in + tid;       // reassign pointer variable (p = expr)
            out[tid] = *p;
        } else {
            out[tid] = 0;
        }
    }
}

// Pointer-to-pointer reassignment
__global__ void ptr_reassign(float **rows, float *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *row = nullptr;
        row = buf + tid * 16;
        if (row != nullptr) {
            for (int j = 0; j < 16; j++)
                row[j] = (float)(tid * 16 + j);
        }
    }
}

// NULL (not nullptr) comparison
__global__ void null_check(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = NULL;
        if (in != NULL)
            p = in + tid;
        out[tid] = (p != NULL) ? *p : -1;
    }
}
