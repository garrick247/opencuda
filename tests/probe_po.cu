// Probe: function-like #define macros, sizeof, global __device__ vars,
// array parameter decay, and atomics on multiple targets.

// ------------------------------------------------------------------
// Function-like #define macros.

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(v, lo, hi) MAX(MIN((v), (hi)), (lo))
#define SQ(x) ((x) * (x))

__global__ void macro_funcs(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        out[tid] = CLAMP(SQ(v), -100, 100);
    }
}

// ------------------------------------------------------------------
// sizeof in expressions.

__global__ void sizeof_expr(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof(int) = 4, sizeof(float) = 4, sizeof(double) = 8
        int a = sizeof(int);
        int b = sizeof(float);
        int c = sizeof(double);
        out[tid] = a + b + c;  // should be 16
    }
}

// ------------------------------------------------------------------
// Global __device__ variable (read-only scalar).

__device__ int g_scale = 3;

__global__ void global_device_var(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid] * g_scale;
    }
}

// ------------------------------------------------------------------
// Array parameter as pointer decay: void fn(int arr[], int n).
// In C/CUDA, `int arr[]` is the same as `int *arr`.

__device__ int arr_sum(int *arr, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += arr[i];
    return s;
}

__global__ void array_param_pointer(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = arr_sum(data, n);
    }
}

// ------------------------------------------------------------------
// Multiple atomic operations in a single kernel.

__global__ void multi_atomic(int *sum, int *cnt, int *mx, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        atomicAdd(sum, v);
        atomicAdd(cnt, 1);
        atomicMax(mx, v);
    }
}

// ------------------------------------------------------------------
// Nested function-like macros with side effects avoided.
// Tests macro expansion with multiple uses of arguments.

#define ABS_DIFF(a, b) ((a) > (b) ? (a) - (b) : (b) - (a))

__global__ void abs_diff_macro(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ABS_DIFF(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// #define with numeric and string arguments (string only in printf).

#define BLOCK_SIZE 128
#define WARP_SIZE  32

__global__ void define_constants(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    if (tid < n && lane_id < BLOCK_SIZE / WARP_SIZE) {
        out[tid] = data[tid] + warp_id;
    }
}
