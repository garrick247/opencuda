// Probe: for-loop continue after multiple variable mutations,
// continue in outer of nested for-loops (inner fully exits),
// continue updates visible in next iteration's condition,
// multiple independent counters surviving continue

// Multiple counters, each conditionally incremented before continue
__global__ void multi_counter_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 0, c = 0;
        for (int i = 0; i < n; i++) {
            int v = in[i];
            if (v % 3 == 0) { a++; continue; }
            if (v % 3 == 1) { b++; continue; }
            c++;
        }
        out[0] = a;
        out[1] = b;
        out[2] = c;
    }
}

// Running sum with early continue — sum tracks correctly across iterations
__global__ void running_sum_continue(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] == 0) continue;  // skip zeros, sum unchanged
            sum += in[i];
            count++;
        }
        out[0] = sum;
        out[1] = count;
    }
}

// Continue in outer loop: inner loop finishes normally, outer continues
__global__ void nested_outer_continue(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        int skipped = 0;
        for (int i = 0; i < n; i++) {
            if (i % 4 == 0) {
                skipped++;
                continue;  // skip entire inner loop for this i
            }
            for (int j = 0; j < 4; j++) {
                total += i * j;
            }
        }
        out[0] = total;
        out[1] = skipped;
    }
}

// continue updates the loop-carried variable that drives the condition
__global__ void stride_continue(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int result = 0;
        int step = 1;
        for (int i = 0; i < n; i += step) {
            if (i % 2 == 0) {
                step = 2;   // modify step before continue
                continue;
            }
            result += i;
            step = 1;
        }
        *out = result;
    }
}
