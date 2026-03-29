// Probe: optimizer × inlining interactions.
// CSE and LICM after device function inlining, multiple calls to same
// device fn, inlined fn in loop body, and complex folding patterns.

// ------------------------------------------------------------------
// Same device function called twice in same expression.
// CSE must NOT merge the two calls if args differ.

__device__ int triple(int x) { return x * 3; }

__global__ void two_calls_diff_args(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = triple(data[tid]);
        int b = triple(data[tid] + 1);
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Device function call in loop body: result used outside loop.
// Tests that inlined call's output survives loop writeback.

__device__ int add_offset(int x, int off) { return x + off; }

__global__ void call_in_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += add_offset(data[i], i);
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Inlined fn result used in constant fold.
// triple(4) should fold to 12 after inlining.

__device__ int triple2(int x) { return x * 3; }

__global__ void const_fold_after_inline(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int offset = triple2(4);  // should fold to 12
        out[tid] = data[tid] + offset;
    }
}

// ------------------------------------------------------------------
// Two device functions, one calling the other (chain).
// Tests that the outer call correctly chains to inner result.

__device__ int inner_fn(int x) { return x * 2 + 1; }
__device__ int outer_fn(int x) { return inner_fn(x) + inner_fn(x + 1); }

__global__ void chained_fn_calls(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = outer_fn(data[tid]);
    }
}

// ------------------------------------------------------------------
// Inlined function with early exit via multiple-return pattern.
// Device fn has conditional return — tests block ordering after inline.

__device__ int clamped(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__global__ void clamp_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = clamped(data[tid], -100, 100);
    }
}

// ------------------------------------------------------------------
// Loop-invariant device fn call: fn(const_arg) in loop.
// The call should be LICM-hoisted (or constant-folded after inlining).

__device__ int is_odd(int x) { return x & 1; }

__global__ void loop_invariant_call(int *out, int *data, int n, int k) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int odd_k = is_odd(k);  // k is loop-invariant
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (is_odd(data[i]) == odd_k) sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Device function with struct parameter — inline with loop.

struct Range {
    int lo, hi;
};

__device__ int in_range(int x, Range r) {
    return (x >= r.lo && x <= r.hi) ? 1 : 0;
}

__global__ void count_in_range(int *out, int *data, int n, int lo, int hi) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Range r;
        r.lo = lo;
        r.hi = hi;
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            cnt += in_range(data[i], r);
        }
        out[0] = cnt;
    }
}
