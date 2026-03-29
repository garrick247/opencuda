// Probe: deeply nested struct field access, struct-with-array members,
// multi-level device function call chains, and pointer-to-struct patterns.

// ------------------------------------------------------------------
// Deeply nested struct: a.b.c field chain.

struct Inner {
    int x;
    int y;
};

struct Outer {
    int tag;
    struct Inner inner;
};

__device__ int sum_outer(struct Outer o) {
    return o.tag + o.inner.x + o.inner.y;
}

__global__ void nested_struct_read(int *out, int *tags, int *xs, int *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Outer o;
        o.tag = tags[tid];
        o.inner.x = xs[tid];
        o.inner.y = ys[tid];
        out[tid] = sum_outer(o);
    }
}

// ------------------------------------------------------------------
// Device functions calling other device functions (3-deep chain).

__device__ int level3(int v) {
    return v * v + 1;
}

__device__ int level2(int v) {
    return level3(v) + level3(v + 1);
}

__device__ int level1(int v) {
    return level2(v) + level2(v - 1);
}

__global__ void deep_call_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = level1(in[tid]);
    }
}

// ------------------------------------------------------------------
// Struct with array member — access arr.data[i].

struct Buf4 {
    int data[4];
    int len;
};

__device__ int buf_sum(struct Buf4 b) {
    int s = 0;
    for (int i = 0; i < b.len; i++) {
        s += b.data[i];
    }
    return s;
}

__global__ void struct_array_member(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Buf4 b;
        b.len = 4;
        b.data[0] = in[tid];
        b.data[1] = in[tid] + 1;
        b.data[2] = in[tid] + 2;
        b.data[3] = in[tid] + 3;
        out[tid] = buf_sum(b);
    }
}

// ------------------------------------------------------------------
// Pointer-to-struct: device fn receives ptr, reads fields.

struct Vec2 {
    float x;
    float y;
};

__device__ float vec2_len_sq(struct Vec2 *v) {
    return v->x * v->x + v->y * v->y;
}

__global__ void ptr_to_struct(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec2 v;
        v.x = xs[tid];
        v.y = ys[tid];
        out[tid] = vec2_len_sq(&v);
    }
}

// ------------------------------------------------------------------
// Mutually sequential calls with struct passed and returned.

struct Stats {
    float sum;
    float sum_sq;
    int count;
};

__device__ struct Stats make_stats(float v) {
    struct Stats s;
    s.sum = v;
    s.sum_sq = v * v;
    s.count = 1;
    return s;
}

__device__ struct Stats merge_stats(struct Stats a, struct Stats b) {
    struct Stats r;
    r.sum = a.sum + b.sum;
    r.sum_sq = a.sum_sq + b.sum_sq;
    r.count = a.count + b.count;
    return r;
}

__global__ void stats_reduction(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid + n < 2 * n) {
        struct Stats sa = make_stats(in[tid]);
        struct Stats sb = make_stats(in[tid + 1 < n ? tid + 1 : tid]);
        struct Stats sc = merge_stats(sa, sb);
        out[tid * 3 + 0] = sc.sum;
        out[tid * 3 + 1] = sc.sum_sq;
        out[tid * 3 + 2] = (float)sc.count;
    }
}

// ------------------------------------------------------------------
// Three-level struct nesting with assignment at each level.

struct Leaf {
    int val;
};

struct Branch {
    struct Leaf left;
    struct Leaf right;
    int weight;
};

struct Tree {
    struct Branch lo;
    struct Branch hi;
};

__global__ void triple_nested_struct(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        struct Tree t;
        t.lo.left.val  = v;
        t.lo.right.val = v + 1;
        t.lo.weight    = 1;
        t.hi.left.val  = v + 2;
        t.hi.right.val = v + 3;
        t.hi.weight    = 2;
        // Combine all leaf values and weights
        int r = t.lo.left.val + t.lo.right.val + t.lo.weight
              + t.hi.left.val + t.hi.right.val + t.hi.weight;
        out[tid] = r;
    }
}
