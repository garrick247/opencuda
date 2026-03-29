// Probe: cross-struct field copy + accumulator patterns
// - acc.x = src.x (copy field from one struct to another's field)
// - acc.x += src.x (compound accumulation)
// - result.field = fn_return.field (cross-return field copy)
// - Struct fields as array indices: arr[s.idx]
// - Multiple struct local vars in same scope

struct V2 { float x, y; };
struct Box { float lo, hi; int n; };

// Copy field from one struct to another
__global__ void field_copy_cross(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        V2 a; a.x = 0.0f; a.y = 0.0f;
        V2 b; b.x = 0.0f; b.y = 0.0f;
        for (int i = 0; i < n; i++) {
            a.x = in[i*2];
            a.y = in[i*2+1];
            b.x = a.x;      // cross-field copy: b.x = a.x
            b.y = a.y;
            out[i*2]   = b.x;
            out[i*2+1] = b.y;
        }
    }
}

// Compound accumulation across struct fields
__global__ void compound_accum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Box box;
        box.lo = 1e9f; box.hi = -1e9f; box.n = 0;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v < box.lo) box.lo = v;
            if (v > box.hi) box.hi = v;
            box.n++;
        }
        out[0] = box.lo; out[1] = box.hi; out[2] = (float)box.n;
    }
}

// Struct field as array index: arr[s.idx]
__global__ void field_as_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Box box;
        box.lo = 0.0f; box.hi = 0.0f; box.n = 0;
        int acc = 0;
        while (box.n < n) {
            acc += in[box.n];  // in[s.idx]
            box.n++;
        }
        out[0] = acc;
    }
}
