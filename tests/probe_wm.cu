// Probe: struct field compound assignment (+=, -=, *=), void* cast to typed ptr,
// multi-level pointer indirection, sizeof in expressions, and
// complex initializer patterns.

// ------------------------------------------------------------------
// Struct field compound assignment operators.

struct Vec { float x, y, z; };

__global__ void struct_compound_assign(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec v;
        v.x = in[tid];
        v.y = in[tid] * 2.0f;
        v.z = in[tid] * 3.0f;

        v.x += 1.0f;         // compound assign to struct field
        v.y -= 0.5f;
        v.z *= 2.0f;
        v.x /= 2.0f;

        out[tid] = v.x + v.y + v.z;
    }
}

// ------------------------------------------------------------------
// Array element compound assignment.

__global__ void array_compound_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int arr[4];
        arr[0] = in[tid];
        arr[1] = in[tid];
        arr[2] = in[tid];
        arr[3] = in[tid];

        arr[0] += 10;
        arr[1] -= 5;
        arr[2] *= 3;
        arr[3] >>= 1;

        out[tid] = arr[0] + arr[1] + arr[2] + arr[3];
        // (v+10) + (v-5) + (3v) + (v/2) [integer] = 5v + v/2 + 5
    }
}

// ------------------------------------------------------------------
// void* cast pattern: store typed data through void*, cast back.

__global__ void void_ptr_cast(int *out, void *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = (int *)buf;
        p[tid] = tid * 2;
        out[tid] = p[tid];
    }
}

// ------------------------------------------------------------------
// Double pointer: int **pp, read and write through it.

__global__ void double_ptr_write(int *out, int **pp, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = pp[tid];
        *p = tid + 1;
        out[tid] = *p;
    }
}

// ------------------------------------------------------------------
// sizeof in expression used as array size (local array).

__global__ void sizeof_expr_array(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int arr[4 * sizeof(int)];   // arr[16] — 4 * 4 bytes
        for (int i = 0; i < 4 * (int)sizeof(int); i++) {
            arr[i] = i;
        }
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += arr[i];
        out[tid] = sum;   // 0+1+...+15 = 120
    }
}

// ------------------------------------------------------------------
// Compound assignment on global array element.

__global__ void global_elem_compound(int *arr, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        arr[tid] += tid;    // compound assign to global memory
        arr[tid] *= 2;
    }
}

// ------------------------------------------------------------------
// Compound bitwise assignments.

__global__ void bitwise_compound(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        v &= 0xFF;    // mask lower byte
        v |= 0x100;   // set bit 8
        v ^= 0x55;    // XOR
        v <<= 2;      // shift left 2
        v >>= 1;      // shift right 1
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Cast void* from global memory (simulating polymorphic buffer).

struct Header { int type; int size; };

__global__ void typed_header_read(int *out, void *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Header *h = (struct Header *)buf + tid;
        out[tid] = h->type + h->size;
    }
}

// ------------------------------------------------------------------
// Three-level indirection: int ***ppp (uncommon but valid C).

__global__ void triple_ptr(int *out, int ***ppp, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int **pp = ppp[tid];
        int *p   = pp[0];
        int v    = *p;
        out[tid] = v;
    }
}
