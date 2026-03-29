// Probe: pointer casts, reinterpret patterns, and unusual type usage.
// (int*)(float*), byte-level access, unsigned pointer arithmetic.

// ------------------------------------------------------------------
// Reinterpret float as int via pointer cast.
// Classic bit manipulation: get sign bit of float.

__global__ void float_bits(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        // Access float bits via int pointer cast
        int bits = *((int*)&v);
        out[tid] = (bits >> 31) & 1;  // sign bit
    }
}

// ------------------------------------------------------------------
// Byte-level access: read first byte of each int.
// Cast int* to unsigned char* for byte access.

__global__ void int_first_byte(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Read low byte of each int
        unsigned char lo = *((unsigned char*)(&data[tid]));
        out[tid] = (int)lo;
    }
}

// ------------------------------------------------------------------
// Unsigned pointer arithmetic with large indices.
// Tests that unsigned index doesn't sign-extend incorrectly.

__global__ void unsigned_index(int *out, int *data, unsigned int n) {
    unsigned int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid] + (int)tid;
    }
}

// ------------------------------------------------------------------
// Null pointer check pattern (checking for 0).

__global__ void null_check(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Simulate null check with 0 comparison
        int v = (data != 0) ? data[tid] : 0;
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Long long pointer arithmetic.
// Stride by 8 bytes (size of long long).

__global__ void ll_stride(long long *out, long long *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid] + (long long)tid;
    }
}

// ------------------------------------------------------------------
// Size-based branching: different behavior for 32/64 bit data.

__global__ void size_branch(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int elem_size = sizeof(int);  // = 4
        int word_idx  = (tid * elem_size) / sizeof(int);  // = tid
        out[tid] = data[word_idx];
    }
}

// ------------------------------------------------------------------
// Array of function-result-sized elements.
// sizeof in array index computation.

__global__ void sizeof_stride(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sizeof(float) = 4; every 4-byte slot holds one float
        int slots = sizeof(float) / sizeof(float);  // = 1
        out[tid * slots] = data[tid * slots] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Mixed signed/unsigned index in loop.
// `for (unsigned i = 0; i < (unsigned)n; i++)` — tests uint loop var.

__global__ void unsigned_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (unsigned int i = 0; i < (unsigned int)n; i++) {
            sum += data[i];
        }
        out[0] = sum;
    }
}
