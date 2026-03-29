// Probe: complex control flow — labeled break patterns (simulate via flags),
// do-while with early return,
// multiple exit paths from a for loop,
// loop with compound condition (&&, ||)

__global__ void find_first(int *out, int *arr, int target, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found = -1;
        for (int i = 0; i < n; i++) {
            if (arr[i] == target) {
                found = i;
                break;
            }
        }
        *out = found;
    }
}

// Do-while loop
__global__ void do_while_sum(int *out, int *arr, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && n > 0) {
        int sum = 0;
        int i = 0;
        do {
            sum += arr[i];
            i++;
        } while (i < n);
        *out = sum;
    }
}

// Loop with compound condition
__global__ void compound_cond(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int sum = 0;
        while (i < n && a[i] > 0) {
            sum += a[i] + b[i];
            i++;
        }
        out[tid] = sum;
    }
}

// Multiple continue statements
__global__ void skip_negatives(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (in[i] < 0.0f) continue;
            if (in[i] > 1000.0f) continue;
            sum += in[i];
        }
        *out = sum;
    }
}
