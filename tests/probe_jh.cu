// Probe: short-circuit && / || in conditions,
// boolean comparison result used as integer,
// prefix ++/-- in loop condition,
// chained relational conditions

// && short-circuit: second operand should not execute when first is false
__global__ void and_guard(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        // Both conditions must be true; second only meaningful when first holds
        if (v >= 0 && v < 100) {
            result = v * 2;
        }
        out[tid] = result;
    }
}

// || short-circuit: result is 1 if either condition holds
__global__ void or_flag(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        if (v < 0 || v >= 100) {
            result = -1;
        } else {
            result = v;
        }
        out[tid] = result;
    }
}

// Boolean comparison result used as integer (0 or 1)
__global__ void bool_to_int(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int is_positive = (v > 0);      // 0 or 1
        int is_even = (v % 2 == 0);     // 0 or 1
        out[tid] = is_positive + is_even * 2;
    }
}

// Prefix --count in while condition: while (--count > 0)
__global__ void countdown(int *out, int start) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = start;
        int sum = 0;
        while (--count > 0) {
            sum += count;
        }
        *out = sum;
    }
}

// Chained && with three predicates
__global__ void triple_and(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        int result = 0;
        if (av > 0 && bv > 0 && cv > 0) {
            result = av + bv + cv;
        }
        out[tid] = result;
    }
}
