// Probe: struct return from __device__ fn, __ldg on int types,
// type widening in mixed arithmetic, conditional store (predicated).

// ------------------------------------------------------------------
// __device__ function returning a struct by value.
// Tests that the return fields are materialized and accessible via .field.

struct MinMax { int lo; int hi; };

__device__ MinMax find_minmax(int *data, int n) {
    MinMax r;
    r.lo = data[0];
    r.hi = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < r.lo) r.lo = data[i];
        if (data[i] > r.hi) r.hi = data[i];
    }
    return r;
}

__global__ void minmax_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        MinMax mm = find_minmax(data, n);
        out[0] = mm.lo;
        out[1] = mm.hi;
    }
}

// ------------------------------------------------------------------
// __ldg on int: ld.global.nc.s32.
// Tests that __ldg works for non-float types.

__global__ void ldg_int(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = __ldg(&data[tid]) + 1;
    }
}

// ------------------------------------------------------------------
// Mixed int/float arithmetic: implicit float promotion.
// int * float should promote int to float and emit mul.f32.

__global__ void int_float_mix(float *out, int *counts, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // int counts[tid] promoted to float for multiplication
        out[tid] = (float)counts[tid] * scale;
    }
}

// ------------------------------------------------------------------
// Conditional store via predicate: if (cond) out[i] = val.
// Tests that the PTX uses @pred bra correctly to skip the store.

__global__ void cond_store(int *out, int *data, int *mask, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (mask[tid] != 0) {
            out[tid] = data[tid];
        }
    }
}

// ------------------------------------------------------------------
// Long chain of struct field mutations — tests no writeback corruption
// across many consecutive field assignments.

struct State6 {
    int a; int b; int c; int d; int e; int f;
};

__global__ void field_chain(int *out, int seed) {
    int tid = threadIdx.x;
    if (tid == 0) {
        State6 s;
        s.a = seed;
        s.b = s.a + 1;
        s.c = s.b + 2;
        s.d = s.c + 3;
        s.e = s.d + 4;
        s.f = s.e + 5;
        out[0] = s.a + s.b + s.c + s.d + s.e + s.f;
    }
}
