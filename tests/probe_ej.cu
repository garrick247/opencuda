// Probe: Unusual struct layouts and access patterns
// - Union-style type punning (parse-only)
// - Struct with padding (how we compute offsets)
// - Struct with a single field
// - Struct with bitfield declaration (parse, ignore bitfield size)
// - Struct used only in pointer context (never instantiated directly)

struct SingleField {
    float value;
};

__device__ float get_val(SingleField sf) {
    return sf.value;
}

__global__ void single_field_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        SingleField sf;
        sf.value = in[tid];
        out[tid] = get_val(sf) * 2.0f;
    }
}

// Struct pointer used as opaque handle (only arrow access)
struct Handle {
    int id;
    float data[4];
    int flags;
};

__global__ void opaque_handle(float *out, Handle *handles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Handle *h = &handles[tid];
        if (h->flags & 1) {
            out[tid] = h->data[0] + h->data[1];
        } else {
            out[tid] = h->data[2] + h->data[3];
        }
    }
}
