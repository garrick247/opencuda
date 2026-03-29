// Probe: tricky C expressions — comma in return, ternary as function arg,
// nested assignment in condition (while((ch = expr) != 0)),
// function call result used as array index, post-increment used in expression

__global__ void post_incr_in_expr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int sum = 0;
        while (i < n) {
            sum += in[i++];  // post-increment as array index
        }
        out[tid] = sum;
    }
}

// Call result as array index
__device__ int hash_fn(int v, int n) {
    return ((v * 2654435761) >> 16) % n;
}

__global__ void hash_scatter(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int h = hash_fn(in[tid], n);
        out[h] = in[tid];
    }
}

// Compound expression in while condition
__global__ void process_while(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0, sum = 0, v;
        while (i < n && (v = in[i]) != -1) {
            sum += v;
            i++;
        }
        out[tid] = sum;
    }
}

// Ternary with side effects as function argument
__global__ void ternary_arg(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // ternary result passed directly to a computation
        int result = (v > 0 ? v : -v) * 2 + (v == 0 ? 1 : 0);
        out[tid] = result;
    }
}
