// Probe: do-while loops, goto labels, multi-variable declarations,
// prefix vs postfix ++/--, comma expressions in for-loops.

// ------------------------------------------------------------------
// Basic do-while: executes body at least once.

__global__ void do_while_basic(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int i = 0;
        do {
            acc += v;
            i++;
        } while (i < 4);
        out[tid] = acc;  // 4*v
    }
}

// ------------------------------------------------------------------
// do-while with break condition inside body.

__global__ void do_while_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0;
        int i = 0;
        do {
            sum += i;
            i++;
            if (sum >= v) break;
        } while (i < 16);
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// do-while counting down.

__global__ void do_while_countdown(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int cnt = 5;
        int result = 0;
        do {
            result += cnt;
            cnt--;
        } while (cnt > 0);
        out[tid] = result;  // 5+4+3+2+1 = 15
    }
}

// ------------------------------------------------------------------
// Multi-variable declaration: int a = 1, b = 2, c = 3.

__global__ void multi_var_decl(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 1, b = 2, c = 3;
        int x = a + b * c;  // 1 + 6 = 7
        out[tid] = x;
    }
}

// ------------------------------------------------------------------
// Multi-variable decl with expressions depending on earlier vars.

__global__ void multi_var_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 1, b = a * 2, c = b - 1;
        out[tid] = c;  // (v+1)*2 - 1 = 2v+1
    }
}

// ------------------------------------------------------------------
// Prefix vs postfix ++ in expressions.

__global__ void prefix_postfix(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v;
        int b = a++;   // b = v, a = v+1
        int c = ++a;   // a = v+2, c = v+2
        out[tid] = b + c;  // v + (v+2) = 2v+2
    }
}

// ------------------------------------------------------------------
// Prefix vs postfix -- in expressions.

__global__ void prefix_postfix_dec(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v;
        int b = a--;   // b = v, a = v-1
        int c = --a;   // a = v-2, c = v-2
        out[tid] = b + c;  // v + (v-2) = 2v-2
    }
}

// ------------------------------------------------------------------
// Comma in for-loop: two update expressions.

__global__ void for_comma_update(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int j = 10;
        for (int i = 0; i < 5; i++, j--) {
            sum += i + j;  // (0+10)+(1+9)+(2+8)+(3+7)+(4+6) = 50
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// goto: simple forward jump.

__global__ void goto_skip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = v;
        if (v > 100) goto done;
        result = result * 2;
done:
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// goto: retry loop pattern (goto backward).

__global__ void goto_retry(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 15;   // 0..15
        int count = 0;
retry:
        if (v > 0) {
            v--;
            count++;
            goto retry;
        }
        out[tid] = count;  // == original (v & 15)
    }
}

// ------------------------------------------------------------------
// do-while with continue.

__global__ void do_while_continue(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        int i = 0;
        do {
            i++;
            if (i % 2 == 0) continue;  // skip even
            acc += i;
        } while (i < 8);
        // odd values 1,3,5,7 → 16
        out[tid] = acc;
    }
}
