// Probe: struct with pointer-typed fields accessed via arrow syntax.
// ptr->pfield is a pointer field — loading it yields another pointer.
// ptr->pfield[i] then indexes into that second pointer.

struct Span {
    float *data;
    int len;
};

// Device function that sums a Span.
__device__ float span_sum(Span *sp) {
    float s = 0.0f;
    for (int i = 0; i < sp->len; i++) {
        s += sp->data[i];
    }
    return s;
}

__global__ void ptr_field_index(float *out, float *arr, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Span sp;
        sp.data = arr;
        sp.len  = n;
        out[0]  = span_sum(&sp);
    }
}

// ------------------------------------------------------------------
// Pointer field written via arrow on an output struct.
// pp->result[tid] = value; where result is a float * field.
// This tests store through a loaded pointer field.

struct OutBuf {
    float *result;
    int    count;
};

__global__ void scatter_write(OutBuf *pp, float *src, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        pp->result[tid] = src[tid] * 2.0f;
    }
}
