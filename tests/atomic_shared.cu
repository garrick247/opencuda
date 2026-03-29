// Regression: atomicAdd on __shared__ memory must emit atom.shared.add,
// NOT atom.global.add. PTX rejects atom.global on shared pointers.
__global__ void shared_histogram(int *out, int *in, int n) {
    __shared__ int hist[16];
    int tid = threadIdx.x;
    // Initialize shared histogram
    if (tid < 16) {
        hist[tid] = 0;
    }
    __syncthreads();

    // Accumulate into shared histogram (must use atom.shared.add)
    if (tid < n) {
        int bucket = in[tid] % 16;
        atomicAdd(&hist[bucket], 1);
    }
    __syncthreads();

    // Write out
    if (tid < 16) {
        out[tid] = hist[tid];
    }
}
