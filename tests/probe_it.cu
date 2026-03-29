// Probe: loop condition variable also modified in body (convergence loop),
// multiple variables modified in complex patterns,
// variables used in increment that were also modified in body

// Standard Newton iteration: both x and f(x) updated each iteration
__global__ void newton_sqrt(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];
        float x = a * 0.5f;  // initial guess
        // 5 Newton iterations for sqrt
        for (int iter = 0; iter < 5; iter++) {
            x = 0.5f * (x + a / x);
        }
        out[tid] = x;
    }
}

// Prefix sum with running accumulator
__global__ void prefix_sum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += in[i];
            out[i] = acc;
        }
    }
}

// Two pointers narrowing from both ends
__global__ void two_ptr_sum(int *result, int *arr, int target, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int lo = 0, hi = n - 1;
        int found = 0;
        while (lo < hi) {
            int s = arr[lo] + arr[hi];
            if (s == target) {
                found = 1;
                break;
            } else if (s < target) {
                lo++;
            } else {
                hi--;
            }
        }
        result[0] = found;
        result[1] = lo;
        result[2] = hi;
    }
}

// Loop where body produces new loop bounds
__global__ void dynamic_bound(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int limit = n;
        int sum = 0;
        for (int i = 0; i < limit; i++) {
            sum += in[i];
            if (sum > 1000) {
                limit = i + 1;  // shrink loop bound mid-loop
            }
        }
        out[0] = sum;
        out[1] = limit;
    }
}
