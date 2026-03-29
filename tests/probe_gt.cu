// Probe: deeply nested do-while, for-in-for-in-while, switch with fallthrough

__global__ void nested_loops_deep(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int total = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                do {
                    int k = (i * 4 + j + tid) % n;
                    total += in[k];
                    k = (k + 1) % n;
                } while (total < 0);  // unlikely but valid
            }
        }
        out[tid] = total;
    }
}

// Switch with fallthrough (no break between cases)
__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 3;
        int result = 0;
        switch (v) {
            case 0:
                result += 1;
                // fallthrough
            case 1:
                result += 2;
                // fallthrough
            case 2:
                result += 4;
                break;
            case 3:
                result = 100;
                break;
        }
        out[tid] = result;
    }
}

// While inside for inside while
__global__ void while_for_while(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        int outer = 0;
        while (outer < 2) {
            for (int i = outer; i < n; i += 2) {
                float v = in[i];
                int inner = 0;
                while (inner < 3) {
                    acc += v * (float)(inner + 1);
                    inner++;
                }
            }
            outer++;
        }
        out[tid] = acc;
    }
}
