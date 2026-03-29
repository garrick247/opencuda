// Probe: Function-like macros that expand to blocks/statements
// - Macro that expands to if statement
// - Macro used in condition of while/for
// - Macro with semicolon inside (statement macro)
// - Macro that expands to expression in declaration initializer

#define CHECK_BOUND(i, n) ((i) >= 0 && (i) < (n))
#define SWAP(a, b, tmp) do { tmp = a; a = b; b = tmp; } while(0)
#define CLAMP_IDX(i, n) ((i) < 0 ? 0 : ((i) >= (n) ? (n)-1 : (i)))

__global__ void macro_stmts(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (CHECK_BOUND(tid, n)) {
        int a = in[tid];
        int b = (tid + 1 < n) ? in[tid + 1] : in[0];
        int tmp;
        SWAP(a, b, tmp);
        int idx = CLAMP_IDX(a - 1, n);
        out[tid] = in[idx] + b;
    }
}

// Macro in loop condition
__global__ void macro_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; CHECK_BOUND(i, n) && i < 8; i++) {
            sum += in[(tid + i) % n];
        }
        out[tid] = sum;
    }
}
