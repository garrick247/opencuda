// Probe: reverse for-loop (count down with i--),
// switch fall-through (no break in one case),
// chained assignment (a = b = 0),
// two-variable for-loop (i++, j-- in increment part)

// Reverse loop: sum elements from n-1 down to 0
__global__ void reverse_sum(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = n - 1; i >= 0; i--) {
            sum += in[i];
        }
        *out = sum;
    }
}

// Switch with intentional fall-through: case 1 falls into case 2
__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        switch (v % 3) {
            case 0:
                result = 10;
                break;
            case 1:
                result = 1;   // fall through into case 2
            case 2:
                result += 100;
                break;
            default:
                result = -1;
        }
        out[tid] = result;
    }
}

// Chained assignment: a = b = 0 should zero both
__global__ void chained_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 99, b = 99;
        a = b = 0;          // chained: b=0, then a=(result of b=0)=0
        // after: a==0, b==0
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
        }
        out[0] = a;         // should be 0
        out[1] = b;         // should be 0
        out[2] = sum;       // sum of in[]
    }
}

// Two-pointer walk: i from left, j from right, swap until they meet
// Uses i++ and j-- in separate statements inside loop
__global__ void two_ptr_walk(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int j = n - 1;
        int neg_count = 0;
        int pos_count = 0;
        while (i <= j) {
            if (in[i] < 0) {
                neg_count++;
                i++;
            } else if (in[j] >= 0) {
                pos_count++;
                j--;
            } else {
                i++;
                j--;
            }
        }
        out[0] = neg_count;
        out[1] = pos_count;
    }
}
