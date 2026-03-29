// Probe: postfix ++/-- on struct fields
// - s.count++ in a loop body (no intervening inline)
// - s.count++ AFTER an inline struct assignment (tricky: _variables["s_count"] has aliased name)
// - s.count-- decrement on a field
// - struct field used as loop counter: it.pos++ in while

struct Stats { float sum; int count; };
struct Iter  { int pos, limit; };

__device__ Stats add_sample(Stats s, float v) {
    Stats r;
    r.sum   = s.sum + v;
    r.count = s.count + 1;
    return r;
}

// Case 1: bare s.count++ in loop, no inline call
__global__ void bare_field_inc(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.sum = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            s.sum += 1.0f;
            s.count++;
        }
        out[0] = s.count;
    }
}

// Case 2: s.count++ AFTER an inline struct assignment
__global__ void field_inc_after_inline(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.sum = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            s = add_sample(s, in[i]);
            s.count++;                 // may silently drop if _stmt_lhs_name not set
        }
        out[0] = s.count;  // expected: 2*n
    }
}

// Case 3: s.count-- decrement
__global__ void field_dec(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s;
        s.sum = 0.0f; s.count = n;
        for (int i = 0; i < n; i++) {
            s.count--;
        }
        out[0] = s.count;  // expected: 0
    }
}

// Case 4: Iter struct with pos++ driving a while loop
__global__ void iter_struct(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Iter it;
        it.pos = 0; it.limit = n;
        int acc = 0;
        while (it.pos < it.limit) {
            acc += in[it.pos];
            it.pos++;
        }
        out[0] = acc;
    }
}
