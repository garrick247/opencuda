// Probe: compound assignments that update loop-carried state,
// += on a variable that feeds the next loop condition,
// loop where the condition uses a value modified via +=

// Loop where sum is updated via +=, read as condition
__global__ void sum_until_thousand(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int count = 0;
        for (int i = 0; i < n && sum < 1000; i++) {
            sum += in[i];
            count++;
        }
        out[0] = sum;
        out[1] = count;
    }
}

// Accumulate with *= (product)
__global__ void running_product(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int prod = 1;
        for (int i = 0; i < n; i++) {
            prod *= in[i];
        }
        *out = prod;
    }
}

// Bitwise &= to mask bits each iteration
__global__ void running_and(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int mask = 0xFFFFFFFFu;
        for (int i = 0; i < n; i++) {
            mask &= in[i];
        }
        *out = mask;
    }
}

// Compound <<= to build a packed bit value
__global__ void pack_bits(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned int packed = 0u;
        for (int i = 0; i < n && i < 32; i++) {
            packed <<= 1;
            packed |= (in[i] & 1u);
        }
        *out = packed;
    }
}
