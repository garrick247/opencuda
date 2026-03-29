// Probe: const/restrict qualifiers, static local, multi-dim arrays,
// array of structs on the stack, and complex initializers.

struct Vec3 {
    float x, y, z;
};

// ------------------------------------------------------------------
// const local variable (should be treated same as non-const in PTX).

__global__ void const_local(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const float scale = 2.5f;
        const int offset = 7;
        out[tid] = in[tid] * scale + (float)offset;
    }
}

// ------------------------------------------------------------------
// __restrict__ pointer (hint only — no semantic change in OpenCUDA).

__global__ void restrict_add(float * __restrict__ out,
                              const float * __restrict__ a,
                              const float * __restrict__ b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

// ------------------------------------------------------------------
// Unsigned comparison (avoid signed-overflow issues).

__global__ void unsigned_cmp(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        unsigned int r = 0;
        for (unsigned int i = 0; i < v && i < 16u; i++) {
            r += i;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Long long arithmetic.

__global__ void longlong_arith(long long *out, long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        long long a = v * 1000000LL;
        long long b = a / 7LL;
        long long c = a % 7LL;
        out[tid] = b + c;
    }
}

// ------------------------------------------------------------------
// Unsigned long long.

__global__ void ulonglong_ops(unsigned long long *out,
                               unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        unsigned long long mask = 0xFFFFFFFFULL;
        unsigned long long lo = v & mask;
        unsigned long long hi = (v >> 32) & mask;
        out[tid] = lo ^ hi;
    }
}

// ------------------------------------------------------------------
// struct Vec3 operations on the stack.

__global__ void vec3_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3 v;
        v.x = in[tid * 3 + 0];
        v.y = in[tid * 3 + 1];
        v.z = in[tid * 3 + 2];
        float dot = v.x * v.x + v.y * v.y + v.z * v.z;
        out[tid] = dot;
    }
}

// ------------------------------------------------------------------
// Integer promotion in mixed arithmetic.

__global__ void int_promotion(int *out, short *sa, char *ca, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // short and char are promoted to int in arithmetic
        int a = sa[tid];    // sign-extended
        int b = ca[tid];    // sign-extended
        int r = a * b + a - b;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Float division by zero → INF (valid in IEEE 754).

__global__ void float_div_zero(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Division by zero produces INF; isfinite check avoids NaN use
        float r = (v != 0.0f) ? (1.0f / v) : 0.0f;
        out[tid] = r;
    }
}
