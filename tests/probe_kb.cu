// Probe: LICM + CSE interaction — invariant that's also a CSE target,
// conditional that changes value mid-loop (CSE must not merge),
// pointer arithmetic with unsigned 64-bit offset,
// while-loop with complex multi-condition exit

// Loop-invariant expression used multiple times: (a * b) is invariant,
// computed in two separate places inside the loop body
__global__ void licm_cse_combined(int *out, int *in, int a, int b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int factor = a * b;    // invariant, must be hoisted or folded
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            sum += v * factor;
            // factor used twice -- CSE must not eliminate the first use
            if (v > factor) {
                sum += factor;
            }
        }
        *out = sum;
    }
}

// CSE must NOT merge two loads from different indices
__global__ void no_cse_different_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n - 1) {
        // Two loads from adjacent locations — must both happen
        int v0 = in[tid];
        int v1 = in[tid + 1];
        out[tid] = v0 + v1;
    }
}

// Pointer cast to char* and byte-level access (uint8 load)
__global__ void byte_access(int *out, unsigned char *bytes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char b = bytes[tid];
        out[tid] = (int)b;
    }
}

// While with two-condition exit: exits if i >= n OR sum > limit
__global__ void while_two_exit(int *out, int *in, int n, int limit) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        while (i < n && sum <= limit) {
            sum += in[i];
            i++;
        }
        out[0] = sum;
        out[1] = i;
    }
}
