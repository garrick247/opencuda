// Probe: goto statements, complex mixed-type arithmetic, early-return
// patterns in nested functions, and verifier stress.

// ------------------------------------------------------------------
// goto: forward and backward jumps.

__global__ void goto_forward(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        if (v < 0) goto negative;
        if (v > 100) goto overflow;
        out[tid] = v;
        goto done;
    negative:
        out[tid] = -1;
        goto done;
    overflow:
        out[tid] = 100;
    done:;
    }
}

// ------------------------------------------------------------------
// Early return in nested if: tests CFG correctness.

__device__ int classify(int x) {
    if (x < 0) return -1;
    if (x == 0) return 0;
    if (x < 10) return 1;
    if (x < 100) return 2;
    return 3;
}

__global__ void classify_kernel(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = classify(data[tid]);
    }
}

// ------------------------------------------------------------------
// Mixed-type arithmetic: int op float → float, with explicit casts.

__global__ void mixed_arith(float *out, int *idata, float *fdata, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   iv = idata[tid];
        float fv = fdata[tid];
        // C rules: int + float → float
        float r1 = iv + fv;
        // int * float → float
        float r2 = iv * fv;
        // int / float → float
        float r3 = (float)iv / fv;
        out[tid] = r1 + r2 + r3;
    }
}

// ------------------------------------------------------------------
// Chained assignments.

__global__ void chain_assign(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = b = c = data[tid] * 2;
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Array of function pointers (not available in GPU code) —
// use a dispatch table with integer codes instead.

__device__ int dispatch(int code, int x) {
    if (code == 0) return x * 2;
    if (code == 1) return x + 10;
    if (code == 2) return x * x;
    return x;
}

__global__ void dispatch_kernel(int *out, int *data, int *codes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = dispatch(codes[tid] % 4, data[tid]);
    }
}

// ------------------------------------------------------------------
// Pre/post increment/decrement: verify ++ and -- in expressions.

__global__ void incr_decr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int a = v++;    // a = v, then v++
        int b = ++v;    // v++, then b = v
        int c = v--;    // c = v, then v--
        int d = --v;    // v--, then d = v
        out[tid] = a + b + c + d + v;
    }
}

// ------------------------------------------------------------------
// Complex boolean: De Morgan's law patterns.

__global__ void demorgan(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid], cv = c[tid];
        // !(A && B) ≡ (!A || !B)
        int r1 = !(av > 0 && bv > 0);
        int r2 = (!(av > 0)) || (!(bv > 0));
        // !(A || B || C) ≡ (!A && !B && !C)
        int r3 = !(av > 0 || bv > 0 || cv > 0);
        int r4 = (!(av > 0)) && (!(bv > 0)) && (!(cv > 0));
        out[tid] = (r1 == r2 ? 1 : 0) + (r3 == r4 ? 1 : 0);
    }
}

// ------------------------------------------------------------------
// Pointer to local: take address of local variable, pass to device fn.

__device__ void fill_two(int *p, int val) {
    p[0] = val;
    p[1] = val + 1;
}

__global__ void local_ptr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[2];
        fill_two(buf, data[tid]);
        out[tid] = buf[0] + buf[1];
    }
}
