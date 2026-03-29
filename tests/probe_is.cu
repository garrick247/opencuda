// Probe: continue in for-loop after body modifications,
// continue in nested loops, break in nested while inside for,
// return from inside nested control flow

// For-loop continue: count and skip
__global__ void for_continue_count(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int pos_count = 0;
        int neg_count = 0;
        int pos_sum = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] < 0) {
                neg_count++;
                continue;
            }
            pos_count++;
            pos_sum += in[i];
        }
        out[0] = pos_count;
        out[1] = neg_count;
        out[2] = pos_sum;
    }
}

// while-loop continue after multiple modifications
__global__ void while_continue_multi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum_odd = 0;
        int sum_even = 0;
        int count = 0;
        while (i < n) {
            int v = in[i];
            i++;
            count++;
            if (v == 0) continue;  // skip zeros, but i/count already updated
            if (v % 2 == 0) {
                sum_even += v;
            } else {
                sum_odd += v;
            }
        }
        out[0] = sum_odd;
        out[1] = sum_even;
        out[2] = count;
    }
}

// Break from while inside for — only breaks inner
__global__ void break_inner_while(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            int j = i;
            while (j < n) {
                if (j > i + 3) break;  // break inner while
                total += j;
                j++;
            }
        }
        *out = total;
    }
}

// Return inside nested control flow
__device__ int first_ge(int *arr, int val, int n) {
    for (int i = 0; i < n; i++) {
        if (arr[i] >= val) {
            return i;
        }
    }
    return n;
}

__global__ void use_first_ge(int *out, int *arr, int *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = first_ge(arr, vals[tid], n);
    }
}
