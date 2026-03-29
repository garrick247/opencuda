// Probe: continue inside doubly-nested if inside for-loop,
// multiple variables modified at different nesting depths before continue,
// if-else chain where only one branch continues

// Doubly-nested if, innermost path continues
__global__ void deep_if_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 0, c = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v > 0) {
                b++;                // always incremented when v > 0
                if (v > 100) {
                    a++;            // incremented before inner continue
                    continue;       // both a and b must be written back
                }
                c++;
            }
        }
        out[0] = a;
        out[1] = b;
        out[2] = c;
    }
}

// Three levels of if nesting with continue at the deepest
__global__ void triple_nested_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int x = 0, y = 0, z = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v != 0) {
                x++;
                if (v > 0) {
                    y++;
                    if (v > 50) {
                        z++;
                        continue;   // x, y, z all modified before continue
                    }
                }
            }
        }
        out[0] = x;
        out[1] = y;
        out[2] = z;
    }
}

// Continue after if-else: the "else" path doesn't continue
__global__ void selective_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int skip_count = 0;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v < 0) {
                skip_count++;   // this branch continues
                continue;
            } else {
                sum += v;       // this branch does NOT continue
            }
        }
        out[0] = sum;
        out[1] = skip_count;
    }
}

// Continue after compound assignment at multiple depths
__global__ void compound_then_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        int count = 0;
        int max_val = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            total += v;   // always updated
            count++;      // always updated
            if (v > max_val) {
                max_val = v;   // update max
                continue;     // skip rest: total and count already updated
            }
            // Non-max path: just continues to next iteration naturally
        }
        out[0] = total;
        out[1] = count;
        out[2] = max_val;
    }
}
