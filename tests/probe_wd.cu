// Probe: chained struct field from function return, cast-of-ternary,
// 3-deep device call chain, do-while(0) macro, assignment in while
// condition, and multiple return paths via struct.

// ------------------------------------------------------------------
// Struct returned from function, field accessed inline.

struct Vec3 { float x, y, z; };

__device__ struct Vec3 make_vec3(float x, float y, float z) {
    struct Vec3 v;
    v.x = x; v.y = y; v.z = z;
    return v;
}

__global__ void struct_field_from_fn(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // Call function, then use field
        struct Vec3 r = make_vec3(v, v * 2.0f, v * 3.0f);
        out[tid] = r.x + r.y + r.z;  // v + 2v + 3v = 6v
    }
}

// ------------------------------------------------------------------
// Cast of ternary result.

__global__ void cast_ternary(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        float r = (float)(v > 0 ? v : -v);   // cast applied to ternary result
        out[tid] = r;  // fabsf(v)
    }
}

// ------------------------------------------------------------------
// 3-deep device call chain.

__device__ int level3(int v) { return v + 1; }
__device__ int level2(int v) { return level3(v) * 2; }
__device__ int level1(int v) { return level2(v) + level3(v); }

__global__ void three_deep_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = level1(in[tid]);
        // level3(v) = v+1
        // level2(v) = 2*(v+1)
        // level1(v) = 2*(v+1) + (v+1) = 3*(v+1) = 3v+3
    }
}

// ------------------------------------------------------------------
// do-while(0) macro idiom (single-pass body, always executes once).

#define SAFE_OP(out, v) do { \
    int _tmp = (v) > 0 ? (v) : 0; \
    *(out) = _tmp * 2;             \
} while (0)

__global__ void do_while_macro(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        SAFE_OP(&out[tid], in[tid]);   // out[tid] = max(in[tid],0)*2
    }
}

// ------------------------------------------------------------------
// Assignment in while condition — while ((v = expr) != sentinel).

__global__ void assign_in_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = tid;
        int sum = 0;
        int v;
        // Accumulate until we hit a zero or exhaust 8 elements
        int i = 0;
        while (i < 8 && (v = in[idx % n]) != 0) {
            sum += v;
            idx++;
            i++;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Multiple early returns from device function.

__device__ int classify(int v) {
    if (v < 0) return -1;
    if (v == 0) return 0;
    if (v < 10) return 1;
    if (v < 100) return 2;
    return 3;
}

__global__ void multi_return_classify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(in[tid]);
    }
}

// ------------------------------------------------------------------
// Nested function calls as arguments.

__device__ int add3(int a, int b, int c) { return a + b + c; }
__device__ int mul2(int a, int b) { return a * b; }

__global__ void nested_call_args(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Nested calls as arguments to add3
        int r = add3(mul2(v, 2), mul2(v, 3), mul2(v, 4));
        // = 2v + 3v + 4v = 9v
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Conditional call: device function called only in one branch.

__device__ int expensive(int v) { return v * v * v; }

__global__ void conditional_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v % 2 == 0) {
            r = expensive(v);   // only called for even v
        } else {
            r = v;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// for-loop with empty init and empty update (only condition).

__global__ void for_bare_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0;
        for (; i < 4;) {
            v += i;
            i++;
        }
        out[tid] = v;  // in[tid] + 0+1+2+3 = in[tid] + 6
    }
}
