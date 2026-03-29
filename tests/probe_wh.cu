// Probe: pre/post-increment in array subscripts, struct array pointer
// iteration, scope-reuse of variable names, and complex loop patterns.

// ------------------------------------------------------------------
// Pre-increment in array subscript: out[++i] — uses incremented value.

__global__ void pre_inc_subscript(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid + 1 < n) {
        int i = tid;
        out[++i] = in[tid];  // out[tid+1] = in[tid]
        out[tid] = i;         // out[tid] = tid+1 (writes the incremented i)
    }
}

// ------------------------------------------------------------------
// Post-increment in array subscript: out[i++] — uses original value.

__global__ void post_inc_subscript(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = 0;
        // Fill 4 consecutive output slots using post-increment
        if (tid * 4 + 3 < n) {
            out[tid * 4 + i++] = in[tid];        // out[tid*4+0], i=1
            out[tid * 4 + i++] = in[tid] * 2;   // out[tid*4+1], i=2
            out[tid * 4 + i++] = in[tid] * 3;   // out[tid*4+2], i=3
            out[tid * 4 + i++] = in[tid] * 4;   // out[tid*4+3], i=4
        }
    }
}

// ------------------------------------------------------------------
// Variable name reuse: inner scope shadows outer.

__global__ void scope_shadow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        {
            int v = v * 2;  // shadows outer v — inner v = 2*outer_v
            r = v;
        }
        // Outer v is still original in[tid]
        out[tid] = r + v;   // 2*in[tid] + in[tid] = 3*in[tid]
    }
}

// ------------------------------------------------------------------
// Struct array iteration with pointer increment.

struct Item { int key; int val; };

__global__ void struct_ptr_iter(int *out, struct Item *items, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Item *p = items + tid * 4;
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += p[i].key + p[i].val;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Combined pre/post in complex expression: (a++) + (++b).

__global__ void mixed_inc(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = in[tid];
        int r = (a++) + (++b);  // (in[tid]) + (in[tid]+1) = 2*in[tid]+1
        // After: a = in[tid]+1, b = in[tid]+1
        out[tid] = r + a + b;   // (2v+1) + (v+1) + (v+1) = 4v+3
    }
}

// ------------------------------------------------------------------
// Loop variable declared in for-init, used after loop (C behavior).

__global__ void loop_var_scope(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int last = -1;
        for (int i = 0; i < 4 && i < n; i++) {
            last = in[i];
        }
        out[tid] = last;   // = in[min(3, n-1)] for tid=any
    }
}

// ------------------------------------------------------------------
// Complex assignment: x = y = z = expr (right-to-left associativity).

__global__ void chained_assignment(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        a = b = c = in[tid] * 2;   // all get same value
        out[tid] = a + b + c;       // 6 * in[tid]
    }
}

// ------------------------------------------------------------------
// Assignment combined with comparison: (x = y) != 0 pattern.

__global__ void assign_cmp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v;
        int r = ((v = in[tid]) != 0) ? v : -1;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Pointer to local struct, accessed via arrow.

__global__ void local_struct_arrow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Item item;
        item.key = in[tid];
        item.val = in[tid] * 2;
        struct Item *p = &item;
        out[tid] = p->key + p->val;  // in[tid] + 2*in[tid] = 3*in[tid]
    }
}

// ------------------------------------------------------------------
// While with multiple decrements.

__global__ void multi_decrement(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 15;   // 0..15
        int count = 0;
        while (v-- > 0) {   // post-decrement: check original v, then decrement
            count++;
        }
        out[tid] = count;   // = original (v & 15)
    }
}
