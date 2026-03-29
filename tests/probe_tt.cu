// Probe: goto patterns, labeled loops (gcc extension-style), and
// complex control flow with multiple labeled targets.

// ------------------------------------------------------------------
// goto forward jump.

__global__ void goto_forward(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        if (v < 0) goto done;
        r = v * 2;
        if (v > 100) goto done;
        r = r + 1;
    done:
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// goto inside nested if.

__global__ void goto_nested(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = 0;
        if (v > 0) {
            a = v;
            if (v > 50) {
                goto skip_b;
            }
            b = v * 2;
        skip_b:;
        }
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// goto used as loop continue (like the goto in probe_sw.cu).

__global__ void goto_outer_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int total = 0;
        for (int i = 0; i < 4; i++) {
            int sub = 0;
            for (int j = 0; j < 4; j++) {
                if (v * i + j > 10) goto next;
                sub += v * i + j;
            }
            total += sub;
            continue;
        next:
            total += sub + 100;
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Multiple labels in the same function.

__global__ void multi_label(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = v;
        if (v < -100) goto neg_large;
        if (v < 0)    goto neg_small;
        if (v > 100)  goto pos_large;
        r = 0;
        goto done;
    neg_large:
        r = -2;
        goto done;
    neg_small:
        r = -1;
        goto done;
    pos_large:
        r = 2;
    done:
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// goto out of nested loop.

__global__ void goto_double_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int found = -1;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (v * i + j == 42) {
                    found = i * 8 + j;
                    goto search_done;
                }
            }
        }
    search_done:
        out[tid] = found;
    }
}
