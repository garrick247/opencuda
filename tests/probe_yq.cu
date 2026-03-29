// Probe: remaining corner cases — __restrict__ with aliasing analysis,
// pointer comparison (p1 < p2), cast between pointer types (int* → char*),
// function pointer (should fail gracefully or be parsed), array decay to
// pointer in function call, compound literal struct init (__device__),
// do-while with complex break, negative modulo behavior, and
// __builtin_expect with zero prediction.

// ------------------------------------------------------------------
// Pointer comparison (ptrdiff / ordering).

__global__ void ptr_compare(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *pa = a + tid;
        int *pb = b + tid;
        // Pointer comparison: implementation-defined but common in CUDA code
        // (both point to global memory, so ordering is meaningful)
        out[tid] = (pa < pb) ? -1 : (pa > pb) ? 1 : 0;
    }
}

// ------------------------------------------------------------------
// Cast int* to char* (byte access).

__global__ void int_as_bytes(unsigned char *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned char *p = (unsigned char *)&v;
        // Extract bytes (little-endian)
        out[tid * 4    ] = p[0];
        out[tid * 4 + 1] = p[1];
        out[tid * 4 + 2] = p[2];
        out[tid * 4 + 3] = p[3];
    }
}

// ------------------------------------------------------------------
// Array decay to pointer (pass array name to function expecting pointer).

__device__ int sum_ptr(int *arr, int len) {
    int s = 0;
    for (int i = 0; i < len; i++) s += arr[i];
    return s;
}

__global__ void array_decay(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int arr[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        // arr decays to int* in the function call
        out[tid] = sum_ptr(arr, 8);
    }
}

// ------------------------------------------------------------------
// Negative modulo behavior (C99: result has sign of dividend).

__global__ void neg_modulo(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        if (y == 0) y = 1;  // avoid div by zero
        int r = x % y;
        // In C99: -7 % 3 == -1, 7 % -3 == 1
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// do-while with complex multi-level break (outer-loop break from inner-if).

__global__ void dowhile_complex_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int steps = 0;
        do {
            int rem = v % 10;
            if (rem == 0) break;  // break out of do-while
            if (rem == 5) { v = v / 5; steps += 10; continue; }
            v = v * rem;
            steps++;
        } while (steps < 20);
        out[tid] = steps;
    }
}

// ------------------------------------------------------------------
// __builtin_expect(cond, 0) — prediction unlikely.

__global__ void builtin_expect_zero(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (__builtin_expect(v < 0, 0)) {
            out[tid] = -v;
        } else {
            out[tid] = v;
        }
    }
}

// ------------------------------------------------------------------
// Volatile pointer to shared memory (explicit volatile qualifier on ptr).

__global__ void volatile_smem_ptr(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    volatile int *vp = (volatile int *)smem;  // volatile pointer to shared
    if (tid < n) vp[tid] = in[tid];
    __syncthreads();
    if (tid < n) out[tid] = vp[tid] * 2;
}

// ------------------------------------------------------------------
// Multiple typedef aliases for the same underlying type.

typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;
typedef float f32;
typedef double f64;

__global__ void typedef_aliases(i32 *out_i, f32 *out_f,
                                   i32 *in_i, f32 *in_f, i32 n) {
    i32 tid = threadIdx.x;
    if (tid < n) {
        i32 iv = in_i[tid];
        f32 fv = in_f[tid];
        out_i[tid] = iv * 2;
        out_f[tid] = fv * 2.0f;
    }
}

// ------------------------------------------------------------------
// Struct with explicit zero-value initializer in middle.

struct ThreeField { int a; float b; int c; };

__device__ float three_sum(struct ThreeField t) {
    return (float)t.a + t.b + (float)t.c;
}

__global__ void three_field_kernel(float *out, int *ia, float *fb, int *ic, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct ThreeField t;
        t.a = ia[tid];
        t.b = fb[tid];
        t.c = ic[tid];
        out[tid] = three_sum(t);
    }
}
