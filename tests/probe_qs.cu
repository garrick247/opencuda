// Probe: many loop-carried variables (register pressure), mixed-type
// phi nodes, post-loop use of loop vars, and loop index reuse patterns.

// ------------------------------------------------------------------
// Many parallel accumulators: tests register allocation under pressure.

__global__ void multi_accum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
        for (int i = 0; i < n; i += 8) {
            s0 += in[i + 0];
            s1 += in[i + 1];
            s2 += in[i + 2];
            s3 += in[i + 3];
            s4 += in[i + 4];
            s5 += in[i + 5];
            s6 += in[i + 6];
            s7 += in[i + 7];
        }
        out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
        out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
    }
}

// ------------------------------------------------------------------
// Loop with post-loop use of loop variable.

__global__ void find_first(int *out, int *data, int target, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        for (; i < n; i++) {
            if (data[i] == target) break;
        }
        out[0] = (i < n) ? i : -1;
    }
}

// ------------------------------------------------------------------
// Nested loop with multiple loop-carried vars.

__global__ void nested_accum(float *out, float *A, float *B, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < n; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n; j++) {
                row_sum += A[i * n + j] * B[j];
            }
            total += row_sum;
        }
        out[0] = total;
    }
}

// ------------------------------------------------------------------
// Mixed-type loop: int index and float accum with type promotions.

__global__ void mixed_loop(float *out, int *keys, float *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum_k = 0;
        float sum_v = 0.0f;
        for (int i = 0; i <= tid; i++) {
            sum_k += keys[i];
            sum_v += vals[i];
        }
        out[tid * 2 + 0] = (float)sum_k;
        out[tid * 2 + 1] = sum_v;
    }
}

// ------------------------------------------------------------------
// Fibonacci-style loop: multiple carry vars depending on each other.

__global__ void fib_mod(int *out, int n, int mod) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            int c = (a + b) % mod;
            a = b;
            b = c;
        }
        out[0] = b;
    }
}

// ------------------------------------------------------------------
// Loop that updates a pointer and a counter together.

__global__ void pack_positive(int *out, int *out_count, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > 0) {
                out[count] = in[i];
                count++;
            }
        }
        out_count[0] = count;
    }
}

// ------------------------------------------------------------------
// Loop with early exit and use of result after.

__global__ void scan_until(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i;
        for (i = 0; i < n; i++) {
            if (sum + data[i] > 1000) break;
            sum += data[i];
        }
        out[0] = sum;
        out[1] = i;
    }
}
