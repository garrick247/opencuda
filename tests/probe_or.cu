// Probe: nested struct with array field, pointer cast patterns,
// conditional assignment patterns, and device fn returning struct ptr.

// ------------------------------------------------------------------
// Nested struct with array field.
// Outer struct contains Inner which has an array field.

struct Ring { float data[4]; int head; int tail; };

__global__ void ring_buf(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Ring r;
        r.head = 0;
        r.tail = 0;
        // Fill ring buffer from input
        for (int i = 0; i < 4 && i < n; i++) {
            r.data[r.tail] = in[i];
            r.tail = (r.tail + 1) & 3;
        }
        // Read out
        for (int i = 0; i < 4; i++) {
            out[i] = r.data[i];
        }
    }
}

// ------------------------------------------------------------------
// Reinterpret cast pattern: float bits as unsigned int.
// Uses bit-cast via pointer reinterpretation (common in GPU code).

__global__ void float_bits(unsigned int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Common pattern: inspect float bit pattern
        float v = data[tid];
        unsigned int bits = *(unsigned int*)&v;  // reinterpret_cast
        out[tid] = bits;
    }
}

// ------------------------------------------------------------------
// Multiple assignments to same variable in different branches.
// Tests that the merge (phi) at the join point has the right value.

__global__ void branch_assign(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int result;
        if (v > 100) {
            result = v - 100;
        } else if (v > 50) {
            result = v - 50;
        } else if (v > 0) {
            result = v;
        } else {
            result = 0;
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Address arithmetic on void*: cast to char* then index.
// Tests handling of void* + offset for generic memory operations.

__global__ void void_ptr_offset(void *out_v, void *in_v, int elem_size, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        char *in  = (char*)in_v  + tid * elem_size;
        char *out = (char*)out_v + tid * elem_size;
        // Copy elem_size bytes
        for (int i = 0; i < elem_size; i++) {
            out[i] = in[i];
        }
    }
}
