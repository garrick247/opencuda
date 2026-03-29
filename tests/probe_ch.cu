// Probe: Patterns at the edge of what the IR supports
// - More than 32 local variables (stress register allocation)
// - Variable declared inside switch case
// - Variable re-declaration in nested block (C99 block scoping)
// - Forward reference to constant defined later

#define NVAR 16

__global__ void many_vars(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v0 = in[tid];
        float v1 = v0 * 1.0f;
        float v2 = v1 + 0.1f;
        float v3 = v2 - 0.1f;
        float v4 = v3 * 1.1f;
        float v5 = v4 / 1.1f;
        float v6 = v5 + v0;
        float v7 = v6 - v1;
        float v8 = v7 * v2;
        float v9 = v8 / (v3 + 1.0f);
        float v10 = v9 + v4;
        float v11 = v10 - v5;
        float v12 = v11 * v6;
        float v13 = v12 / (v7 + 1.0f);
        float v14 = v13 + v8;
        float v15 = v14 - v9;
        out[tid] = v0 + v5 + v10 + v15;
    }
}

// Variable in switch case
__global__ void switch_local_var(int *out, int *sel, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int result;
        switch (sel[tid]) {
            case 0: {
                int local_a = in[tid] * 2;
                result = local_a;
                break;
            }
            case 1: {
                int local_b = in[tid] + 100;
                result = local_b;
                break;
            }
            default:
                result = -1;
                break;
        }
        out[tid] = result;
    }
}
