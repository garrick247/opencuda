// Probe: multi-variable for-loop init (int i=0, j=n-1; ...),
// comma expressions in loop update (i++, j--),
// parallel iteration patterns

__global__ void two_ptr_reverse(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0, j = n - 1;
        while (i < j) {
            float tmp = in[i];
            out[i] = in[j];
            out[j] = tmp;
            i++;
            j--;
        }
        if (i == j) out[i] = in[i];
    }
}

// Two loop variables via sequential declaration + for loop
__global__ void staircase_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        int lo = 0;
        int hi = tid;
        for (int k = lo; k <= hi; k++) {
            s += in[k];
        }
        out[tid] = s;
    }
}

// For loop with two increments (comma operator in update)
__global__ void interleave(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int j = 0;
        int k = 0;
        while (k < n * 2) {
            if (i < n) { out[k++] = a[i++]; }
            if (j < n) { out[k++] = b[j++]; }
        }
    }
}
