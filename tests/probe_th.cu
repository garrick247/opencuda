// Probe: complex lvalue assignments — compound ops on struct fields,
// 2D array element writes, pointer field increments, and early return patterns.

struct Counter { int hits, misses, total; };

// ------------------------------------------------------------------
// Compound assignment on struct fields.

__global__ void struct_compound(struct Counter *ctrs, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > 0) {
            ctrs[tid].hits += 1;
        } else {
            ctrs[tid].misses += 1;
        }
        ctrs[tid].total += 1;
    }
}

// ------------------------------------------------------------------
// 2D array (flattened) with row/col indexing.

__global__ void mat2d_write(int *mat, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        mat[row * cols + col] = row * 100 + col;
    }
}

// ------------------------------------------------------------------
// Early return (guard clause pattern) — no else branch.

__global__ void guard_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int v = in[tid];
    if (v == 0) return;  // guard: skip zeros
    out[tid] = 1000 / v;  // safe because v != 0
}

// ------------------------------------------------------------------
// Multiple early returns.

__global__ void multi_early_return(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int x = a[tid];
    if (x < 0) { out[tid] = -1; return; }
    int y = b[tid];
    if (y < 0) { out[tid] = -2; return; }
    if (x == 0 && y == 0) { out[tid] = 0; return; }
    out[tid] = x * y;
}

// ------------------------------------------------------------------
// Increment/decrement operators on array elements.

__global__ void inc_array(int *arr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        arr[tid]++;
    }
}

__global__ void dec_array(int *arr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        arr[tid]--;
    }
}

// ------------------------------------------------------------------
// Prefix vs postfix increment in expression.

__global__ void pre_post_inc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v++;   // a = v, then v incremented (postfix)
        int b = ++v;   // v incremented, then b = v (prefix)
        out[tid] = a + b;  // a + b = v + (v+2) = 2*original_v + 2
    }
}

// ------------------------------------------------------------------
// Compound assignment on pointer-indexed element.

__global__ void ptr_compound(float *data, float scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = data + tid;
        *p *= scale;
        *p += 1.0f;
    }
}

// ------------------------------------------------------------------
// Struct field compound assignment via arrow.

__global__ void arrow_compound(struct Counter *ctrs, int *deltas, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        ctrs[tid].total += deltas[tid];
        ctrs[tid].hits  -= (deltas[tid] < 0) ? 1 : 0;
    }
}
