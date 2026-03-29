// Probe: LICM boundary (only hoist truly invariant expressions),
// device function called in only one branch,
// break from outer loop of a two-level nest,
// loop start from a local variable

// LICM boundary: scale is invariant, but scaled_v is NOT
__global__ void licm_boundary(int *out, int *in, int scale, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int log2scale = 0;
        int tmp = scale;
        while (tmp > 1) {
            log2scale++;
            tmp >>= 1;   // this loop should NOT hoist the >>= since tmp changes
        }
        // Now log2scale is a shift amount for scale (floor(log2(scale)))
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            int scaled_v = v >> log2scale;  // log2scale is loop-invariant
            sum += scaled_v;
        }
        *out = sum;
    }
}

// Device function called conditionally (in if-branch only)
__device__ int heavy_compute(int v) {
    int r = v;
    r = r * r - r + 1;
    return r;
}

__global__ void conditional_device_call(int *out, int *in, int *mask, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        if (mask[tid]) {
            result = heavy_compute(v);
        } else {
            result = v;
        }
        out[tid] = result;
    }
}

// Break outer loop: search for first row with all positives
__global__ void break_outer_loop(int *out, int *mat, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found_row = -1;
        for (int r = 0; r < rows; r++) {
            int all_pos = 1;
            for (int c = 0; c < cols; c++) {
                if (mat[r * cols + c] <= 0) {
                    all_pos = 0;
                    break;   // break inner c-loop
                }
            }
            if (all_pos) {
                found_row = r;
                break;   // break outer r-loop
            }
        }
        *out = found_row;
    }
}

// Loop starting from a computed local variable
__global__ void loop_from_local(int *out, int *in, int start, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        // start is a parameter but treat like local
        int end = start + n;
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += in[i - start];   // adjust index back to 0-based
        }
        *out = sum;
    }
}
