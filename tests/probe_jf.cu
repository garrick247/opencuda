// Probe: switch statement with fall-through, break, and default,
// switch with variable modified before break,
// nested switch in loop, switch on enum-like constants

// Basic switch with explicit breaks
__global__ void switch_classify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        switch (v % 4) {
            case 0: result = 100; break;
            case 1: result = 200; break;
            case 2: result = 300; break;
            case 3: result = 400; break;
            default: result = -1; break;
        }
        out[tid] = result;
    }
}

// Switch with fall-through (no break between cases)
__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int flags = 0;
        switch (v) {
            case 3: flags |= 4;   // fall through
            case 2: flags |= 2;   // fall through
            case 1: flags |= 1; break;
            default: flags = -1;
        }
        out[tid] = flags;
    }
}

// Switch inside a for loop
__global__ void switch_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int counts[4] = {0, 0, 0, 0};
        for (int i = 0; i < n; i++) {
            switch (in[i] & 3) {
                case 0: counts[0]++; break;
                case 1: counts[1]++; break;
                case 2: counts[2]++; break;
                case 3: counts[3]++; break;
            }
        }
        out[0] = counts[0];
        out[1] = counts[1];
        out[2] = counts[2];
        out[3] = counts[3];
    }
}

// Switch with variable mutation before break
__global__ void switch_mutate(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 0;
        switch (v) {
            case 1: x = 10; x *= 2; break;
            case 2: x = 20; x += 5; break;
            default: x = -1;
        }
        out[tid] = x;
    }
}
