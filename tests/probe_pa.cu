// Probe: optimizer stress — strength reduction edge cases, identity fold
// with casts, CSE across different types, and dead-store elimination.

// ------------------------------------------------------------------
// Strength reduction: x * 2 → x + x, x * 1 → x, x + 0 → x.
// Tests that these don't over-fire and produce wrong results.

__global__ void strength_edge(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int a = v * 1;      // should fold to v
        int b = v * 0;      // should fold to 0
        int c = v + 0;      // should fold to v
        int d = v - 0;      // should fold to v
        int e = 0 + v;      // should fold to v
        int f = v * 2;      // strength reduce → v + v
        int g = v << 0;     // identity → v
        int h = v >> 0;     // identity → v
        // Sum everything so none are DCE'd
        out[tid] = a + b + c + d + e + f + g + h;
        // Expected: v + 0 + v + v + v + 2v + v + v = 8v
    }
}

// ------------------------------------------------------------------
// CSE: same expression computed twice in the same block.
// Both `data[tid] * 3 + 1` computations should collapse to one.

__global__ void cse_double(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int x = v * 3 + 1;
        int y = v * 3 + 1;   // should be CSE'd to same register as x
        out[tid] = x + y;    // should be 2 * (v*3+1)
    }
}

// ------------------------------------------------------------------
// Cast through same type is identity: (int)(int)v → v.
// Double cast shouldn't produce extra cvt instructions.

__global__ void double_cast(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = data[tid];
        int a = (int)(float)f;   // float → float → int: inner cast is nop
        int b = (int)(int)a;     // int → int → int: both casts are nop
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Constant propagation through a chain of assignments.
// All temporaries should be folded to the constant 42.

__global__ void const_chain(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 6;
        int b = a * 7;       // 42
        int c = b - 0;       // 42
        int d = c * 1;       // 42
        int e = 0 + d;       // 42
        out[0] = e;
    }
}

// ------------------------------------------------------------------
// Dead store: variable written but immediately overwritten before any read.
// The first store should be eliminated.

__global__ void dead_store(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = 999;         // dead: immediately overwritten
        v = data[tid];       // live write
        out[tid] = v;
    }
}
