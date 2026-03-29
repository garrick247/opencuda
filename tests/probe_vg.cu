// Probe: array parameter syntax in __global__ kernels, void* pointers,
// complex short-circuit, nested ternary as function argument,
// and predicate-heavy patterns.

// ------------------------------------------------------------------
// __global__ kernel with array[] parameter syntax.

__global__ void array_param_kernel(float out[], float in[], int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Void pointer: cast and use.

__global__ void void_ptr_cast(void *out_v, void *in_v, int n) {
    int tid = threadIdx.x;
    int *out = (int *)out_v;
    int *in  = (int *)in_v;
    if (tid < n) {
        out[tid] = in[tid] + 1;
    }
}

// ------------------------------------------------------------------
// Complex short-circuit: (a && b) || (c && d).

__global__ void complex_short_circuit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v > 0, b = v < 100, c = v > -100, d = v < 0;
        int r = (a && b) ? 1 : ((c && d) ? 2 : 3);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Nested ternary as function argument.

__device__ int triple_add(int x, int y, int z) {
    return x + y + z;
}

__global__ void ternary_as_arg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // All three args are ternaries
        int r = triple_add(
            v > 0  ? v     : -v,
            v > 10 ? v - 5 : v + 5,
            v < 0  ? 0     : 1
        );
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Many predicates alive simultaneously.

__global__ void pred_pressure(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // 8 predicates alive before any use
        int p0 = v > 0;
        int p1 = v > 10;
        int p2 = v > 20;
        int p3 = v > 30;
        int p4 = v < 0;
        int p5 = v < -10;
        int p6 = v < -20;
        int p7 = v < -30;
        // Use all of them
        int r = (p0 + p1 + p2 + p3) - (p4 + p5 + p6 + p7);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Ternary chain: long sequence of ternary expressions.

__global__ void ternary_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // 5-level ternary chain
        int r = v > 100 ? 5
              : v > 50  ? 4
              : v > 0   ? 3
              : v > -50 ? 2
              :           1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Nested function call with side effects (atomicAdd + conditional use).

__global__ void atomic_in_expr(int *out, int *counter, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // atomicAdd has side effect — must not be CSE'd or eliminated
        int c = atomicAdd(counter, 1);
        int r = (c % 2 == 0) ? v * 2 : v + c;
        out[tid] = r;
    }
}
