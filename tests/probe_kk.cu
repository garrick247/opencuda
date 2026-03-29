// Probe: device fn result as array index,
// comparison between two device fn results,
// widening multiply (int→long long before multiply),
// const pointer parameter,
// post-increment as expression subterm

// Device function whose return value is used as an array index
__device__ int hash(int v, int size) {
    return (v * 2654435761u) % size;  // Knuth hash mod size
}

__global__ void hash_scatter(int *out, int *in, int size, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int idx = hash(in[tid], size);   // device fn result → array index
        out[idx] = in[tid];
    }
}

// Compare two device fn results
__device__ int score(int v) {
    return v * v + v + 1;   // some function
}

__global__ void compare_device_results(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Branch on comparison of two device fn calls
        if (score(a[tid]) > score(b[tid])) {
            out[tid] = 1;
        } else if (score(a[tid]) == score(b[tid])) {
            out[tid] = 0;
        } else {
            out[tid] = -1;
        }
    }
}

// Widening multiply to avoid overflow
__global__ void wide_multiply(long long *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long va = (long long)a[tid];
        long long vb = (long long)b[tid];
        out[tid] = va * vb;   // 64-bit product
    }
}

// const pointer parameter: compiler should still emit same loads
__global__ void sum_const_ptr(int *out, const int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += in[i];   // load from const pointer
        }
        *out = total;
    }
}

// Post-increment in array subscript: arr[i++]
// After: old i used as index, i is incremented
__global__ void post_inc_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        while (i < n) {
            sum += in[i++];  // use i as index, then increment
        }
        *out = sum;
    }
}
