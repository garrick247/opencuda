// Probe: struct with array field, namespace-scoped types, int64 arithmetic
// These are areas the memory notes as future fixes.

// ------------------------------------------------------------------
// Struct with small fixed-size array field.
// Access via ptr->arr[i] and s.arr[i] must both work.

struct Hist4 {
    int counts[4];
};

__device__ void hist_add(Hist4 *h, int bucket) {
    h->counts[bucket] += 1;
}

__global__ void struct_array_field(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Hist4 h; h.counts[0] = 0; h.counts[1] = 0; h.counts[2] = 0; h.counts[3] = 0;
        for (int i = 0; i < n; i++) {
            int v = data[i] & 3;  // bucket 0..3
            hist_add(&h, v);
        }
        out[0] = h.counts[0];
        out[1] = h.counts[1];
        out[2] = h.counts[2];
        out[3] = h.counts[3];
    }
}

// ------------------------------------------------------------------
// 64-bit integer arithmetic: long long accumulator.
// Tests that add.s64, mul.lo.s64 etc. are emitted correctly.

__global__ void int64_accum(long long *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        long long sum = 0LL;
        for (int i = 0; i < n; i++) {
            sum += (long long)data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Unsigned long long accumulator.
// Tests that u64 operations use the correct PTX types.

__global__ void uint64_accum(unsigned long long *out, unsigned int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        unsigned long long sum = 0ULL;
        for (int i = 0; i < n; i++) {
            sum += (unsigned long long)data[i];
        }
        out[0] = sum;
    }
}
