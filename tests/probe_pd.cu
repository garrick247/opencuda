// Probe: compound condition in while, comma operator (absent — skip),
// pointer cast chains, bitfield mask patterns, and name collision safety.

// ------------------------------------------------------------------
// Compound condition in while loop: while (i < n && data[i] != -1).
// Tests that the while-loop compound condition is NOT incorrectly unrolled
// and that both parts are evaluated.

__global__ void while_compound(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        while (i < n && data[i] != -1) {
            sum += data[i];
            i++;
        }
        out[0] = sum;
        out[1] = i;
    }
}

// ------------------------------------------------------------------
// Pointer cast chain: cast float* → void* → int* then read bits.
// Tests that multi-hop pointer casts don't lose address space info.

__global__ void ptr_cast_chain(unsigned int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        void *vp = (void*)&v;         // float* → void*
        unsigned int bits = *(unsigned int*)vp;  // void* → uint*
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// Name collision across kernels: local var named `sum` appears in two
// kernels — each should have its own register.

__global__ void name_a(int *out, int *data, int n) {
    int tid = threadIdx.x;
    int sum = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += data[i];
    }
    out[tid] = sum;
}

__global__ void name_b(float *out, float *data, int n) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += data[i];
    }
    out[tid] = sum;
}

// ------------------------------------------------------------------
// Bitfield extraction with signed arithmetic:
// sign-extend 8-bit field from a 32-bit word.

__global__ void sign_extend(int *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int w = data[tid];
        // Extract bits [15:8] as a signed 8-bit value
        int field = (int)((w >> 8) & 0xFF);
        // Sign-extend: if bit 7 is set, fill upper bits with 1s
        if (field & 0x80) field |= ~0xFF;
        out[tid] = field;
    }
}
