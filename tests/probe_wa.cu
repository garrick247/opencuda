// Probe: struct/array aggregate initializers, volatile pointer semantics,
// __all_sync/__any_sync/__ballot_sync, and complex #define with
// side-effect arguments.

// ------------------------------------------------------------------
// Struct initializer: Record r = {expr, expr}.

typedef struct {
    int   x;
    float y;
} Pair;

__global__ void struct_init_brace(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p = {in[tid], (float)in[tid] * 1.5f};
        out[tid] = p.y + (float)p.x;
    }
}

// ------------------------------------------------------------------
// Array initializer: int arr[4] = {1, 2, 3, 4}.

__global__ void array_init_brace(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int lut[4] = {10, 20, 30, 40};
        int idx = tid % 4;
        out[tid] = lut[idx];
    }
}

// ------------------------------------------------------------------
// Partial array initializer: int arr[8] = {1, 2}; rest are zero.

__global__ void array_partial_init(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int arr[8] = {1, 2};  // arr[2..7] = 0
        int sum = 0;
        for (int i = 0; i < 8; i++) sum += arr[i];
        out[tid] = sum;  // 1 + 2 + 0*6 = 3
    }
}

// ------------------------------------------------------------------
// Volatile local variable: must not be optimized away.

__global__ void volatile_local(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        volatile int v = in[tid];
        volatile int w = v + 1;
        out[tid] = w;
    }
}

// ------------------------------------------------------------------
// Volatile global pointer: loads must not be CSE'd.

__global__ void volatile_ptr(int *out, volatile int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];   // must load
        int b = in[tid];   // must also load (volatile prevents CSE)
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// __all_sync: all threads in warp satisfy condition?

__global__ void all_sync_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // All threads in the warp have v > 0?
        int all_pos = __all_sync(0xFFFFFFFF, v > 0);
        out[tid] = all_pos;
    }
}

// ------------------------------------------------------------------
// __any_sync: any thread in warp satisfies condition?

__global__ void any_sync_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int any_neg = __any_sync(0xFFFFFFFF, v < 0);
        out[tid] = any_neg;
    }
}

// ------------------------------------------------------------------
// __ballot_sync: bitmask of which threads satisfy condition.

__global__ void ballot_sync_kernel(unsigned int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int mask = __ballot_sync(0xFFFFFFFF, v > 0);
        out[tid] = mask;
    }
}

// ------------------------------------------------------------------
// Macro with side-effect argument (double evaluation hazard).

#define SQUARE(x) ((x) * (x))

__global__ void macro_side_effect(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // SQUARE(v) — safe, no side effects
        int s = SQUARE(v);
        out[tid] = s;  // v*v
    }
}

// ------------------------------------------------------------------
// Zero-initialized struct.

__global__ void zero_struct(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p = {0, 0.0f};
        // Explicitly zero-init, then conditionally set
        if (tid % 2 == 0) {
            p.x = tid;
            p.y = (float)tid;
        }
        out[tid] = p.x + (int)p.y;  // tid+tid=2*tid for even, 0 for odd
    }
}
