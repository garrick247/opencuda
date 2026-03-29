// Probe: switch fallthrough, do-while with break, ternary as lvalue-ish,
// post-increment in complex positions, and initializer expressions.

// ------------------------------------------------------------------
// Switch with fallthrough (no break between cases).

__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 4;
        int r = 0;
        switch (v) {
            case 0:
                r += 10;
                // fallthrough
            case 1:
                r += 1;
                break;
            case 2:
                r += 2;
                // fallthrough
            case 3:
                r += 3;
                break;
            default:
                r = -1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Do-while with break in the middle.

__global__ void do_while_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        int i = 0;
        do {
            if (v <= 0) break;
            acc += v;
            v--;
            i++;
        } while (i < 5);
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Post-increment used as function argument (value before increment).

__global__ void post_incr_arg(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = tid;
        int b = tid;
        // a++ means use a THEN increment
        int x = a++ + b;   // x = tid + tid = 2*tid, then a = tid+1
        int y = a + b++;   // y = (tid+1) + tid, then b = tid+1
        out[tid] = x + y;  // 2*tid + (2*tid+1) = 4*tid+1
    }
}

// ------------------------------------------------------------------
// Comma operator in while condition.

__global__ void comma_in_while(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0, acc = 0;
        // comma in while: both sides evaluated, last is condition
        while (i++, i <= 5) {
            acc += i;
        }
        out[tid] = acc + tid;
    }
}

// ------------------------------------------------------------------
// Nested ternary.

__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Classify into 4 buckets via nested ternary
        int r = (v < 0)   ? -1 :
                (v < 10)  ?  0 :
                (v < 100) ?  1 : 2;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// While with compound condition and post-increment inside.

__global__ void while_compound_postincr(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        int sum = 0;
        while (i < n && i < 8) {
            sum += in[i++];
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Multiple assignments in single expression (right-associative).

__global__ void chained_assign(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = b = c = tid * 3;
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Conditional expression as statement (result discarded).

__global__ void ternary_side_effect(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        // Ternary used for conditional increment
        v > 0 ? r++ : (r = -1);
        out[tid] = r;
    }
}
