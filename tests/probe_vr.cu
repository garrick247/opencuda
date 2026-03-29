// Probe: CSE safety (load after store must not CSE with pre-store load),
// loops with multiple exit types, deep if-else nesting.

// ------------------------------------------------------------------
// Two loads from same address with intervening store — must NOT CSE.

__global__ void load_store_load(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v1 = in[tid];          // first load
        in[tid] = v1 + 1;          // store (modifies address)
        int v2 = in[tid];          // second load — must reload, not CSE
        out[tid] = v1 + v2;        // (v) + (v+1) = 2v+1
    }
}

// ------------------------------------------------------------------
// Loop with both break and early continue and complex accumulation.

__global__ void loop_multi_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 16; i++) {
            if (i % 3 == 0) continue;     // skip multiples of 3
            if (acc > v) break;            // stop when accumulated > v
            acc += i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Deep if-else-if chain (6 levels).

__global__ void deep_if_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v < 0) r = -1;
        else if (v < 10) r = 0;
        else if (v < 50) r = 1;
        else if (v < 100) r = 2;
        else if (v < 500) r = 3;
        else if (v < 1000) r = 4;
        else r = 5;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Complex loop: multiple loop-carried variables, break on condition.

__global__ void multi_var_loop_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = v, c = 1;
        for (int i = 0; i < 32; i++) {
            a += b;
            b = b / 2 + c;
            c = c * 2;
            if (c > 64) break;  // exits after 7 iterations (c=1,2,4,8,16,32,64,→break)
        }
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Nested loops with continue in outer and break in inner.

__global__ void nested_continue_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            if (i == 4) continue;   // skip outer iteration 4
            for (int j = 0; j < 8; j++) {
                if (j == v % 8) break;  // break inner at position v%8
                acc += i + j;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Load-load CSE within same block is OK (no intervening store).

__global__ void load_cse_ok(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v1 = in[tid];    // first load
        int v2 = in[tid];    // same address, no intervening store — may CSE
        out[tid] = v1 + v2;  // = 2 * in[tid]
    }
}

// ------------------------------------------------------------------
// Return from inside nested loops.

__device__ int find_in_grid(const int *grid, int rows, int cols,
                              int target) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r * cols + c] == target) {
                return r * cols + c;  // return from deep nesting
            }
        }
    }
    return -1;
}

__global__ void grid_search(int *out, int *data, int *targets, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = find_in_grid(data, 4, 4, targets[tid]);
    }
}
