// Probe: complex multi-level control flow — nested loops with break/continue,
// early exit from switch inside loop, and loop-carried accumulation patterns
// that stress the optimizer's per-block constraint.

// ------------------------------------------------------------------
// Nested loop: inner break affects inner only.

__global__ void nested_break(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (j >= v) break;  // inner break
                sum += i * 8 + j;
            }
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Continue in inner loop.

__global__ void nested_continue(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid] % 4;
        int count = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if ((i + j) % 2 == v) continue;
                count++;
            }
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Loop with multiple continues.

__global__ void multi_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 16; i++) {
            if (i % 2 == 0) continue;
            if (i % 3 == 0) continue;
            acc += i * v;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// While loop with multiple breaks.

__global__ void while_multi_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int i = 0, acc = 0;
        while (i < 32) {
            if (v < 0 && i > 8) break;
            if (v > 100 && i > 4) break;
            acc += i;
            i++;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Loop with switch inside.

__global__ void switch_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            switch ((v + i) % 4) {
                case 0: acc += 1; break;
                case 1: acc += 2; break;
                case 2: acc += 4; break;
                default: acc += 8; break;
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Do-while with nested if.

__global__ void do_while_nested(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = v, count = 0;
        do {
            if (x > 0) {
                x -= 3;
            } else if (x < -10) {
                break;
            } else {
                x -= 1;
            }
            count++;
        } while (count < 20);
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Loop-carried dependency: Fibonacci-style recurrence.

__global__ void fib_loop(int *out, int *seeds, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = seeds[tid * 2 + 0];
        int b = seeds[tid * 2 + 1];
        for (int i = 0; i < 10; i++) {
            int c = a + b;
            a = b;
            b = c;
        }
        out[tid] = b;
    }
}

// ------------------------------------------------------------------
// Accumulation with conditional update (loop-carried through predicate).

__global__ void cond_accum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int pos = 0, neg = 0, zero = 0;
        for (int i = 0; i < 8; i++) {
            int val = v * i - 12;
            if (val > 0) pos++;
            else if (val < 0) neg++;
            else zero++;
        }
        out[tid] = pos * 100 + neg * 10 + zero;
    }
}
