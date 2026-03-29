// Probe: pointer arithmetic, arrow chains, array-of-structs access,
// and multi-step address computations.

// ------------------------------------------------------------------
// Arrow operator: struct pointer field access via ->.

struct Node {
    int val;
    int next_idx;
};

__global__ void arrow_field_read(int *out, struct Node *nodes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = nodes[tid].val;
        int ni = nodes[tid].next_idx;
        if (ni >= 0 && ni < n) {
            out[tid] = v + nodes[ni].val;
        } else {
            out[tid] = v;
        }
    }
}

// ------------------------------------------------------------------
// Pointer arithmetic: ptr + N, *(ptr + N).

__global__ void ptr_arith(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = in + tid;
        int v = *p;
        // Access neighbors via pointer arithmetic
        int left  = (tid > 0)     ? *(p - 1) : 0;
        int right = (tid < n - 1) ? *(p + 1) : 0;
        out[tid] = left + v + right;
    }
}

// ------------------------------------------------------------------
// Array of structs: varying field access pattern.

struct Particle {
    float px;
    float py;
    float vx;
    float vy;
    float mass;
};

__global__ void struct_array_fields(float *out, struct Particle *particles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float px = particles[tid].px;
        float py = particles[tid].py;
        float vx = particles[tid].vx;
        float vy = particles[tid].vy;
        float m  = particles[tid].mass;
        // Kinetic energy: 0.5 * m * (vx^2 + vy^2)
        float ke = 0.5f * m * (vx * vx + vy * vy);
        // Potential energy placeholder: sqrt(px^2 + py^2)
        float dist = sqrtf(px * px + py * py);
        out[tid] = ke + dist;
    }
}

// ------------------------------------------------------------------
// Chained pointer increment: manual iteration via pointer walking.

__global__ void ptr_walk(int *out, int *in, int n, int stride) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = in + tid;
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += *p;
            p += stride;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Pointer passed to device fn, modified through the pointer.

__device__ void accumulate_into(int *acc, int val, int weight) {
    *acc += val * weight;
}

__global__ void ptr_out_param(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        accumulate_into(&acc, v,     1);
        accumulate_into(&acc, v + 1, 2);
        accumulate_into(&acc, v + 2, 3);
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Nested array indexing: 2D array via flat pointer + manual index.

__global__ void matrix_diag(float *out, float *mat, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid < rows && tid < cols) {
        // Diagonal element: mat[tid * cols + tid]
        out[tid] = mat[tid * cols + tid];
    }
}

// ------------------------------------------------------------------
// Multiple struct arrays read simultaneously.

struct ColorRGB {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

__global__ void blend_colors(int *out, struct ColorRGB *src, struct ColorRGB *dst,
                              int n, float alpha) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sr = (float)src[tid].r;
        float sg = (float)src[tid].g;
        float sb = (float)src[tid].b;
        float dr = (float)dst[tid].r;
        float dg = (float)dst[tid].g;
        float db = (float)dst[tid].b;
        int rr = (int)(alpha * sr + (1.0f - alpha) * dr);
        int rg = (int)(alpha * sg + (1.0f - alpha) * dg);
        int rb = (int)(alpha * sb + (1.0f - alpha) * db);
        out[tid] = (rr << 16) | (rg << 8) | rb;
    }
}
