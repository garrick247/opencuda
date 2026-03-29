// Probe: deeply nested if-else (5 levels),
// deeply nested loop (3 levels),
// complex mixed control flow with break/continue at different levels,
// large number of live variables simultaneously

// 5-level nested if-else: variable updated at each level
__global__ void deep_if_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        if (v > 0) {
            result += 1;
            if (v > 10) {
                result += 10;
                if (v > 100) {
                    result += 100;
                    if (v > 1000) {
                        result += 1000;
                        if (v > 10000) {
                            result += 10000;
                        }
                    }
                }
            }
        }
        out[tid] = result;
    }
}

// 3-level nested loops with separate accumulators
__global__ void triple_nested_accum(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    total += i * n * n + j * n + k;
                }
            }
        }
        *out = total;
    }
}

// Break from middle loop only (not innermost, not outermost)
__global__ void mid_loop_break(int *out, int *in, int m, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            int row_sum = 0;
            for (int j = 0; j < n; j++) {
                int v = in[i * n + j];
                if (v < 0) break;   // break inner j-loop
                row_sum += v;
            }
            sum += row_sum;
        }
        *out = sum;
    }
}

// Many live variables at once: 8 independent accumulators
__global__ void eight_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            s0 += (v >> 0) & 1;
            s1 += (v >> 1) & 1;
            s2 += (v >> 2) & 1;
            s3 += (v >> 3) & 1;
            s4 += (v >> 4) & 1;
            s5 += (v >> 5) & 1;
            s6 += (v >> 6) & 1;
            s7 += (v >> 7) & 1;
        }
        out[0]=s0; out[1]=s1; out[2]=s2; out[3]=s3;
        out[4]=s4; out[5]=s5; out[6]=s6; out[7]=s7;
    }
}
