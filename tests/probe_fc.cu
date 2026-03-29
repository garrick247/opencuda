// Probe: casting edge cases — cast of cast, cast in array subscript,
// cast of pointer arithmetic result, C-style cast of struct field

__global__ void cast_chain(float *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Cast of cast
        float v = (float)(int)(in[tid] & 0xFF);
        // Cast in array subscript
        int idx = (int)((float)tid * 1.5f) % n;
        out[tid] = v + (float)in[idx];
    }
}

// Integer cast used as bool condition
__global__ void cast_as_cond(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // (int)v used as boolean
        if ((int)v) {
            out[tid] = 1;
        } else {
            out[tid] = 0;
        }
    }
}

// Cast of pointer difference to int
__global__ void ptr_cast(int *out, float *start, float *end, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Pointer arithmetic cast
        int dist = (int)(end - start);
        out[tid] = dist + tid;
    }
}

// Unsigned cast to prevent sign extension
__global__ void unsigned_cast(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Prevent sign extension in shift
        unsigned int uv = (unsigned int)v;
        unsigned int shifted = uv >> 1;
        out[tid] = (int)shifted;
    }
}
