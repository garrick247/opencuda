// Regression: empty statement (null statement ';') in function body
// Without fix: ParseError "unexpected token ';'" when macro expands to nothing
// Fix: _parse_stmt handles lone SEMI token as null statement (C standard §6.8.3)

// Macros that expand to nothing — common in CUDA code for assertions,
// logging, and no-op annotations
#define CUDA_ASSERT(x)
#define UNUSED(x)       (void)(x)
#define LIKELY(x)       (x)
#define NOOP()

__global__ void null_stmt_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    CUDA_ASSERT(tid < n);   // expands to nothing → bare semicolon
    NOOP();                  // empty statement from zero-arg macro
    ;                        // explicit null statement
    if (LIKELY(tid < n)) {
        UNUSED(n);
        out[tid] = in[tid] * 2;
    }
}

__global__ void semicolon_guard(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) { ; return; }  // semicolon inside braces
    for (int i = 0; ; i++) {     // empty for-condition
        if (i >= 4) break;
        out[tid] += in[tid + i];
    }
}
