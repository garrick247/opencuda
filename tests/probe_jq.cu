// Probe: variable declared inside switch case,
// while loop body declares variable with same name as outer,
// do-while loop with inner variable declarations,
// variable declared inside if-true branch only (not else) then used after merge

// Variable declared inside a case: should be local to that case
__global__ void switch_local_var(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        switch (v % 3) {
            case 0: {
                int x = v * 2;
                result = x + 1;
                break;
            }
            case 1: {
                int x = v + 10;  // same name 'x' in different case
                result = x - 1;
                break;
            }
            default:
                result = -v;
                break;
        }
        out[tid] = result;
    }
}

// While loop body declares variable with same name as outer scope
__global__ void while_inner_decl(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int limit = 100;
        int count = 0;
        int i = 0;
        while (i < n) {
            int limit = in[i];  // inner 'limit' shadows outer
            if (limit > 0) {
                count++;
            }
            i++;
        }
        out[0] = count;
        out[1] = limit;  // should be 100, not last in[i]
    }
}

// do-while with inner variable declarations and continue
__global__ void do_while_inner(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        do {
            int v = in[i];    // 'v' declared in do-while body
            int doubled = v * 2;  // 'doubled' declared in do-while body
            sum += doubled;
            i++;
        } while (i < n);
        *out = sum;
    }
}

// Variable declared only in if-true (no else), then read after merge
// After-merge should see: v*2 if condition, 0 otherwise
__global__ void if_only_decl(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        if (v > 0) {
            int doubled = v * 2;  // only declared in if-true
            result = doubled;
        }
        // After merge: result = doubled (if v>0) or 0
        out[tid] = result;
    }
}
