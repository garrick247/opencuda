// Probe: multi-return device functions, nested ternary chains, loop-local struct
// Tests correctness of early-return emulation and complex control flow.

// ------------------------------------------------------------------
// Device function with early return in if-else body.
// The compiler inlines these — both return paths must converge correctly.

__device__ int safe_div(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

__global__ void div_safe(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_div(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Device function with 3 return points (nested if-else if-else).
// Tests that all three merge paths produce the right result.

__device__ int classify(int v) {
    if (v < 0) return -1;
    if (v == 0) return 0;
    return 1;
}

__global__ void sign_classify(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(data[tid]);
    }
}

// ------------------------------------------------------------------
// Chained ternary (nested ternary): (cond1) ? a : (cond2) ? b : c.
// Tests that the phi merge at the outer ternary correctly sees
// the inner ternary result.

__global__ void chain_ternary(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int result = (v < -10) ? -2
                   : (v < 0)   ? -1
                   : (v == 0)  ? 0
                   : (v < 10)  ? 1
                   :              2;
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Struct declared inside a loop body — re-initialized each iteration.
// Tests that struct fields are correctly reset without aliasing.

struct Point2 { int x; int y; };

__global__ void loop_struct_reinit(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            Point2 p;
            p.x = i;
            p.y = i * 2;
            sum += p.x + p.y;
        }
        out[0] = sum;
    }
}
