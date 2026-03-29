// Probe: output-via-pointer device functions, pass-by-ref patterns,
// swap functions, and multi-output device functions.

// ------------------------------------------------------------------
// Device function that writes to an output pointer.

__device__ void square_root_approx(float x, float *result) {
    // Newton's method, 2 iterations
    float r = x * 0.5f;
    r = r - (r * r - x) * 0.5f / r;
    r = r - (r * r - x) * 0.5f / r;
    *result = r;
}

__global__ void sqrt_via_ptr(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float r;
        square_root_approx(data[tid], &r);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Swap function: modifies two values via pointers.

__device__ void swap_ints(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void sort2(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        int a = data[tid * 2];
        int b = data[tid * 2 + 1];
        if (a > b) swap_ints(&a, &b);
        out[tid * 2]     = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// Two-output device function: min and max via pointers.

__device__ void minmax(int a, int b, int *mn, int *mx) {
    *mn = (a < b) ? a : b;
    *mx = (a > b) ? a : b;
}

__global__ void compute_minmax(int *out_mn, int *out_mx,
                                int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int mn, mx;
        minmax(a[tid], b[tid], &mn, &mx);
        out_mn[tid] = mn;
        out_mx[tid] = mx;
    }
}

// ------------------------------------------------------------------
// Accumulate via pointer: device fn adds to running sum.

__device__ void accumulate(int *sum, int val) {
    *sum += val;
}

__global__ void accum_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            accumulate(&total, data[i]);
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Three-level inlining: kernel → a → b → c.
// Tests that chained inlines produce correct register bindings.

__device__ int double_val(int x) {
    return x * 2;
}

__device__ int quad_val(int x) {
    return double_val(double_val(x));
}

__device__ int oct_val(int x) {
    return quad_val(quad_val(x));
}

__global__ void three_level_inline(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = oct_val(data[tid]);  // data[tid] * 8
    }
}

// ------------------------------------------------------------------
// Device fn called with same arg twice (potential CSE interaction).

__device__ int add_self(int x) {
    return x + x;
}

__global__ void self_add_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Called twice with different args — should not share state
        int r1 = add_self(data[tid]);
        int r2 = add_self(data[tid] + 1);
        out[tid] = r1 + r2;
    }
}

// ------------------------------------------------------------------
// Output-pointer device fn called in a loop.
// The pointer argument must be live across loop iterations.

__device__ void running_min(int val, int *cur_min) {
    if (val < *cur_min) *cur_min = val;
}

__global__ void loop_min(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int mn = data[0];
        for (int i = 1; i < n; i++) {
            running_min(data[i], &mn);
        }
        out[0] = mn;
    }
}
