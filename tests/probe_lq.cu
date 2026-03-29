// Probe: loop with type conversion in body (should unroll + chain correctly),
// loop body with multiple dependent computations,
// accumulator using different type than loop var (int loop, float acc),
// loop with early break (runtime condition — should NOT unroll)

// Loop with float accumulation using int loop index
__global__ void float_acc_int_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {   // trip 8, should unroll
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop where body computes using both i and accumulator
__global__ void weighted_sum(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += i * i;   // sum = 0+1+4+9+16+25+36+49 = 140
        }
        out[0] = sum;   // 140
    }
}

// Loop with break based on runtime value — should NOT unroll
__global__ void loop_early_break(int *out, int *in, int n, int limit) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > limit) break;   // runtime break — loop not unrollable
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop with continue based on runtime value
__global__ void loop_with_continue(int *out, int *in, int n, int skip) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] == skip) continue;  // skip specific values
            sum += in[i];
        }
        out[0] = sum;
    }
}

// Loop with CvtInst in body: int index → float computation
__global__ void cvt_in_loop(float *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            sum += (float)i;   // CvtInst from int to float in loop body
        }
        out[0] = sum;   // 0+1+2+3+4+5+6+7 = 28.0f
    }
}
