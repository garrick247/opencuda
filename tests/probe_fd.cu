// Probe: variable shadowing — local var shadows outer scope, loop var shadows
// param, inner block var shadows function-level var

__global__ void var_shadow(float *out, float *in, int n) {
    int tid = threadIdx.x;
    float sum = 0.0f;  // outer sum
    if (tid < n) {
        float sum = in[tid];  // shadows outer sum — inner block
        for (int i = 0; i < 3; i++) {
            float sum = in[(tid + i) % n];  // shadows again
            out[(tid + i) % n] += sum;
        }
        out[tid] = sum;  // inner-if-block sum, not loop sum
    }
    // outer sum still 0 here (dead)
}

// Loop variable shadows outer variable of same name
__global__ void loop_shadow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = in[tid];  // outer i
        for (int i = 0; i < 4; i++) {  // inner i shadows
            out[tid] += in[(tid + i) % n];
        }
        out[tid] += i;  // outer i used after loop
    }
}
