// Probe: two-variable for-loop (int i=0, j=n-1; i<j; i++, j--),
// for-loop with non-trivial increment (i += 2, stride loops),
// multiple assignment targets in increment expression,
// loop variable used both as index and in computation

// Two-pointer convergence pattern: i from left, j from right
__global__ void two_pointer_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0, j = n - 1;
        while (i < j) {
            sum += in[i] + in[j];
            i++;
            j--;
        }
        *out = sum;
    }
}

// Stride-2 loop: i += 2
__global__ void sum_evens(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i += 2) {
            sum += in[i];
        }
        *out = sum;
    }
}

// Loop index used in two different computations in the same body
__global__ void index_squared_plus_index(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += i * i + i;   // i appears twice in body
        }
        *out = sum;
    }
}

// Decrement loop: for (int i = n-1; i >= 0; i--)
__global__ void reverse_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = n - 1; i >= 0; i--) {
            sum += in[i];
        }
        *out = sum;
    }
}

// Nested loop with two independent accumulators
__global__ void dot_products(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int dot = 0, cross = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            cross += a[i] * b[n - 1 - i];
        }
        out[0] = dot;
        out[1] = cross;
    }
}
