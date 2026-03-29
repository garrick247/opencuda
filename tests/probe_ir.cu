// Probe: break/continue interaction with variable writeback,
// break from nested loop (outer only),
// continue in while loop,
// goto-style "labeled break" simulation with flag

__global__ void break_in_nested(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found_i = -1, found_j = -1;
        // Find first (i,j) pair where i*j > n
        for (int i = 1; i <= n; i++) {
            int done = 0;
            for (int j = 1; j <= n; j++) {
                if (i * j > n) {
                    found_i = i;
                    found_j = j;
                    done = 1;
                    break;
                }
            }
            if (done) break;
        }
        out[0] = found_i;
        out[1] = found_j;
    }
}

// Continue in while loop
__global__ void continue_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        while (i < n) {
            i++;
            if (in[i-1] < 0) continue;
            sum += in[i-1];
        }
        *out = sum;
    }
}

// Break with loop-carried value: break should use the value AT the break point
__global__ void break_with_carry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int acc = 0;
        int last = 0;
        for (int i = 0; i < n; i++) {
            acc += in[i];
            last = in[i];
            if (acc > 1000) break;
        }
        out[0] = acc;
        out[1] = last;
    }
}

// Return from inside a loop
__device__ int search(int *arr, int target, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

__global__ void use_search(int *out, int *arr, int target, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        *out = search(arr, target, n);
    }
}
