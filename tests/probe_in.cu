// Probe: for-loop with externally declared loop variable,
// loop variable assigned result of complex expression before loop,
// post-loop variable value usage

__global__ void extern_loop_var(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i;  // declared outside
        for (i = 0; i < n; i++) {
            sum += i;
        }
        out[0] = sum;
        out[1] = i;  // i should be n after loop
    }
}

// Loop var pre-initialized by expression
__global__ void expr_loop_var(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int start = n / 2;
        int sum = 0;
        for (int i = start; i < n; i++) {
            sum += i;
        }
        out[0] = sum;
    }
}

// Nested loops, outer var used in inner
__global__ void nested_loops_dep(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                total += i * j;
            }
        }
        *out = total;
    }
}

// While loop with externally-modified condition variable
__global__ void while_extern_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = in[0];
        int steps = 0;
        while (x != 1 && steps < n) {
            if (x % 2 == 0) {
                x = x / 2;
            } else {
                x = 3 * x + 1;
            }
            steps++;
        }
        out[0] = steps;
        out[1] = x;
    }
}
