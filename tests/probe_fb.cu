// Probe: complex loop patterns — do-while with break, for-loop with multiple
// continues, loop inside switch, switch inside loop

__global__ void loop_switch_interact(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;
        for (int i = 0; i < 8; i++) {
            switch (v & 3) {
                case 0: result += i; break;
                case 1: result -= i; continue;  // continue in switch inside loop
                case 2: result ^= i; break;
                default: result |= i; break;
            }
            v >>= 1;
        }
        out[tid] = result;
    }
}

// Do-while with multiple exit conditions
__global__ void do_while_multi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        do {
            if (v <= 0) break;
            v = v / 2;
            count++;
            if (count > 30) break;
        } while (v > 1);
        out[tid] = count;
    }
}

// For-loop with continue that skips work
__global__ void for_continue(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (i == tid) continue;  // skip self
            if (in[i] < 0.0f) continue;  // skip negatives
            sum += in[i];
        }
        out[tid] = sum;
    }
}
