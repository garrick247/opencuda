// Regression: extern __shared__ float sdata[] — dynamic shared memory
// Without fix: "extern" not in lexer → ParseError "undefined variable 'extern'"
// Fix: 'extern' added to _KEYWORDS (same as 'inline'/'register'), parser handles
//   empty brackets [] as size=0 sentinel → codegen emits .extern .shared .align N .b8 name[];

extern __shared__ float sdata[];

__global__ void reduce_sum(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction
    int s;
    for (s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[bid] = sdata[0];
    }
}

__global__ void scan_shared(int *input, int *output, int n) {
    extern __shared__ int isdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    isdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // Prefix scan
    int offset;
    for (offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid >= offset) {
            isdata[tid] += isdata[tid - offset];
        }
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = isdata[tid];
    }
}
