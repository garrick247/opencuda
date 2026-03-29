// Probe: break + continue coexisting in same loop,
// variable updated before break (included in result) vs after (excluded),
// while-loop with manual increment in different branch positions,
// multiple accumulators with complex break/continue patterns

// Both break and continue in same for-loop
__global__ void break_and_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v == 0) continue;    // skip zeros
            if (v < 0) break;        // stop at first negative
            sum += v;
            count++;
        }
        out[0] = sum;
        out[1] = count;
    }
}

// Variable updated BEFORE break vs AFTER: sum gets break element, extra does not
__global__ void partial_break_update(int *out, int *in, int n, int limit) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int extra = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];           // always updated (even at break iteration)
            if (sum > limit) break;
            extra += in[i];        // NOT updated for the break iteration
        }
        out[0] = sum;
        out[1] = extra;
    }
}

// While-loop: manual increment before continue
__global__ void while_manual_inc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        int skipped = 0;
        while (i < n) {
            if (in[i] < 0) {
                i++;          // increment BEFORE continue (correctly skips)
                skipped++;
                continue;
            }
            sum += in[i];
            i++;
        }
        out[0] = sum;
        out[1] = skipped;
    }
}

// For-loop: break inside deeply nested if, with outer variable updates
__global__ void deep_nested_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 0, c = 0;
        for (int i = 0; i < n; i++) {
            a += 1;             // always increments
            if (in[i] > 0) {
                b += in[i];
                if (in[i] > 100) {
                    c += 1;
                    if (in[i] > 1000) {
                        break;  // break from deep inside nested ifs
                    }
                }
            }
        }
        out[0] = a;
        out[1] = b;
        out[2] = c;
    }
}
