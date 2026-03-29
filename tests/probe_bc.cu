// Probe: unusual control flow
// - early return from nested if inside for loop
// - break from nested for loop (inner only)
// - continue in while loop
// - multiple returns with different value types from __device__ function
// - return inside switch case

__device__ int first_nonzero(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] != 0) return i;
    }
    return -1;
}

__global__ void find_first(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        out[0] = first_nonzero(in, n);
    }
}

__global__ void nested_break(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int found = -1;
        for (int i = 0; i < n && found < 0; i++) {
            for (int j = 0; j < n; j++) {
                if (i * n + j == tid) {
                    found = i * 10 + j;
                    break;  // breaks inner loop only
                }
            }
        }
        out[tid] = found;
    }
}

__global__ void continue_while(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int i = 0;
        while (i < n) {
            i++;
            if (i % 3 == 0) continue;
            sum += i;
        }
        out[tid] = sum;
    }
}
