// Regression: module-level __shared__ with explicit size
// Without fix: ParseError "expected RBRACKET, got INT_LIT '256'"
// Fix: module-level __shared__ handler parses optional size before ']'

#define BLOCK 256

__shared__ float s_data[BLOCK];
__shared__ int s_idx[64];

__global__ void global_shared_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    s_data[tid] = (tid < n) ? in[tid] : 0.0f;
    s_idx[tid % 64] = tid;
    __syncthreads();
    if (tid < n) {
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum += s_data[(tid + i) % BLOCK];
        }
        out[tid] = sum * 0.25f;
    }
}
