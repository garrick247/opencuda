// Probe: optimizer interactions with short-circuit blocks,
// LICM safety across short-circuit, and function call in condition.

// ------------------------------------------------------------------
// Loop-invariant expression in && condition RHS.
// `i < n && factor > 0.0f` — factor is loop-invariant.
// LICM should be safe since factor is never written in the loop.

__global__ void licm_in_and(float *out, float *data, float factor, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n && factor > 0.0f; i++) {
            sum += data[i] * factor;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Device function call in && condition.
// `tid < n && is_valid(data[tid])` — function call guards the load.

__device__ int is_valid(int v) { return v >= 0 && v < 1000; }

__global__ void fn_in_and(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && is_valid(data[tid])) {
        out[tid] = data[tid] * 2;
    } else {
        out[tid] = -1;
    }
}

// ------------------------------------------------------------------
// Multiple && in single expression: a && b && c where b has load.
// `i >= 0 && data[i] >= 0 && data[i] < 100`

__global__ void triple_and(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (i >= 0 && data[i] >= 0 && data[i] < 100) {
                count++;
            }
        }
        out[0] = count;
    }
}

// ------------------------------------------------------------------
// && in ternary condition: (a && b) ? x : y.

__global__ void and_in_ternary(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Use && as the ternary condition
        int result = (v > 0 && v < 100) ? v * 2 : 0;
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// || in while condition with two array accesses.
// `while (i < n && (a[i] != 0 || b[i] != 0))` — load a[i] first, b[i] only if a[i]==0.

__global__ void or_in_while(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int i = 0;
        while (i < n && (a[i] != 0 || b[i] != 0)) {
            count++;
            i++;
        }
        out[0] = count;
    }
}
