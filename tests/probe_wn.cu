// Probe: for(;;) with break, continue inside switch (skips to loop),
// nested switch+break inside for-loop, return from switch, and
// complex loop/switch interaction patterns.

// ------------------------------------------------------------------
// for(;;) infinite loop broken by internal condition.

__global__ void infinite_loop_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 0xFF;   // 0..255
        int count = 0;
        for (;;) {
            if (v == 0) break;
            v = (v * 3 + 1) & 0xFF;   // Collatz-like
            count++;
            if (count >= 32) break;    // safety bound
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// continue in switch: continues the enclosing for-loop.

__global__ void switch_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            switch ((v + i) % 4) {
                case 0: continue;     // skips rest of for-loop body for this i
                case 1: sum += 1; break;
                case 2: sum += 2; break;
                case 3: sum += 3; break;
            }
            sum += 10;    // only reached if case != 0
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// return from inside switch statement.

__device__ int switch_early_return(int v) {
    switch (v % 4) {
        case 0: return 100;
        case 1: return 200;
        case 2: return 300;
        // case 3: fall through to default
        default: return -1;
    }
}

__global__ void switch_return_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = switch_early_return(in[tid]);
    }
}

// ------------------------------------------------------------------
// Nested: switch inside for inside switch.

__global__ void nested_switch_for(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        switch (v % 2) {
            case 0:
                for (int i = 0; i < 4; i++) {
                    switch (i) {
                        case 0: acc += 1; break;
                        case 1: acc += 2; break;
                        default: acc += 3; break;
                    }
                }
                break;  // break outer switch
            case 1:
                acc = v;
                break;
        }
        out[tid] = acc;
        // even: 1+2+3+3=9, odd: v
    }
}

// ------------------------------------------------------------------
// Loop with labeled continue (simulated by early goto in C).
// C doesn't have labeled break/continue — test regular break from inner loop.

__global__ void inner_break_outer_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (j == 2) break;   // only inner break
                acc += i + j;
            }
            // continues outer loop after inner break
        }
        out[tid] = acc;  // each i: j=0(i+0) + j=1(i+1) = 2i+1
        // i=0: 1, i=1: 3, i=2: 5, i=3: 7 → total = 16
    }
}

// ------------------------------------------------------------------
// Complex: switch with fallthrough AND inner loop with break.

__global__ void switch_fallthrough_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 3;
        int acc = 0;
        switch (v) {
            case 0:
                // falls through to case 1
            case 1:
                for (int i = 0; i < 3; i++) {
                    acc += i + 1;   // 1+2+3 = 6
                    if (acc > 4) break;  // breaks after i=1 (acc=3, then i=2: acc=6>4)
                }
                break;  // breaks switch
            case 2:
                acc = 99;
                break;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// for loop with break and else-equivalent pattern.

__global__ void loop_search_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 8;
        int found = -1;
        int keys[8] = {3, 1, 4, 1, 5, 9, 2, 6};
        for (int i = 0; i < 8; i++) {
            if (keys[i] == v) {
                found = i;
                break;
            }
        }
        out[tid] = found;  // first index of v in keys, or -1
    }
}

// ------------------------------------------------------------------
// While loop with decrement and multiple conditions.

__global__ void while_decrement_multi(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 8, b = 3;
        while (a > 0 && b > 0) {
            a -= 2;
            b--;
        }
        out[tid] = a + b;
        // b hits 0 first after 3 iters: a=8-6=2, b=0 → 2
    }
}
