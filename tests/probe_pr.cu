// Probe: pointer arithmetic edge cases.
// Pointer difference, multi-stride indexing, byte-stride pointer,
// pointer cast to intptr, global __device__ pointer array,
// conditional pointer selection.

// ------------------------------------------------------------------
// Stride-2 pointer walk: every other element.
// Tests that pointer arithmetic uses correct element size.

__global__ void stride2_access(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n / 2) {
        out[tid] = data[tid * 2] + data[tid * 2 + 1];
    }
}

// ------------------------------------------------------------------
// Byte-stride pointer: cast to char* for byte-level access.
// Tests that (char*) cast produces byte-stride PTX arithmetic.

__global__ void byte_stride(int *out, unsigned char *bytes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Each element is 4 bytes wide; read first byte of each group
        unsigned char b = bytes[tid * 4];
        out[tid] = (int)b;
    }
}

// ------------------------------------------------------------------
// Pointer-to-pointer: array of pointers (simulated via offsets).
// Each thread uses its block index to pick a sub-array.

__global__ void ptr_from_array(float *out, float *data, int stride, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid < n) {
        // Access element from block's sub-array
        float v = data[bid * stride + tid];
        out[bid * stride + tid] = v * 2.0f;
    }
}

// ------------------------------------------------------------------
// Negative index: pointer arithmetic with negative offset.
// Accesses data[tid - 1] for tid > 0.

__global__ void negative_index(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid > 0 && tid < n) {
        out[tid] = data[tid] - data[tid - 1];
    }
    if (tid == 0) {
        out[0] = data[0];
    }
}

// ------------------------------------------------------------------
// Pointer comparison: find pointer to minimum.
// Tests that pointer-typed variables can be selected conditionally.

__device__ float* ptr_min(float *a, float *b) {
    return (*a < *b) ? a : b;
}

__global__ void pointer_select(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = ptr_min(&a[tid], &b[tid]);
        out[tid] = *p;
    }
}

// ------------------------------------------------------------------
// 2D grid indexing: row * width + col.
// Tests multi-operand pointer arithmetic with two dimensions.

__global__ void grid2d_access(float *out, float *data, int width, int height) {
    int col = threadIdx.x;
    int row = blockIdx.x;
    if (col < width && row < height) {
        float v = data[row * width + col];
        out[row * width + col] = v + (float)(row + col);
    }
}

// ------------------------------------------------------------------
// Pointer with loop-carried offset.
// `ptr += stride` each iteration — offset must survive loop writeback.

__global__ void ptr_walk(float *out, float *data, int stride, int steps) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < steps; i++) {
            sum += data[i * stride];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Conditional pointer update: pointer assigned in if/else.
// Tests that pointer value from both branches is live at merge.

__global__ void cond_ptr_update(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p;
        if (a[tid] > 0) {
            p = b;
        } else {
            p = c;
        }
        out[tid] = p[tid];
    }
}
