// Probe: struct field access patterns — nested indexing, field
// arithmetic, struct pointer arithmetic, and member function patterns.

struct Vec2i { int x, y; };
struct Vec4f { float x, y, z, w; };
struct Matrix2x2 { float a, b, c, d; };  // row-major: [a b; c d]

// ------------------------------------------------------------------
// Struct array indexing with field extraction.

__global__ void extract_fields(float *xout, float *yout,
                                Vec4f *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        xout[tid] = in[tid].x;
        yout[tid] = in[tid].y;
    }
}

// ------------------------------------------------------------------
// Matrix-vector multiply via struct.

__device__ void matvec(Matrix2x2 m, float *vx, float *vy) {
    float nx = m.a * (*vx) + m.b * (*vy);
    float ny = m.c * (*vx) + m.d * (*vy);
    *vx = nx;
    *vy = ny;
}

__global__ void mat_vec_kernel(float *ox, float *oy,
                                Matrix2x2 *mats,
                                float *ix, float *iy, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float vx = ix[tid], vy = iy[tid];
        matvec(mats[tid], &vx, &vy);
        ox[tid] = vx;
        oy[tid] = vy;
    }
}

// ------------------------------------------------------------------
// Struct field used in condition.

__global__ void struct_cond(int *out, Vec2i *pts, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2i p = pts[tid];
        int in_quad1 = (p.x > 0 && p.y > 0);
        int in_quad3 = (p.x < 0 && p.y < 0);
        out[tid] = in_quad1 ? 1 : in_quad3 ? 3 : 0;
    }
}

// ------------------------------------------------------------------
// Accumulate struct fields.

__global__ void accum_struct(float *sumx, float *sumy, Vec4f *pts, int n) {
    __shared__ float sx[256], sy[256];
    int tid = threadIdx.x;
    sx[tid] = (tid < n) ? pts[tid].x : 0.0f;
    sy[tid] = (tid < n) ? pts[tid].y : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sx[tid] += sx[tid + s];
            sy[tid] += sy[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sumx, sx[0]);
        atomicAdd(sumy, sy[0]);
    }
}

// ------------------------------------------------------------------
// Write struct fields from computation.

__global__ void write_struct(Vec4f *out, float *x, float *y, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float xv = x[tid], yv = y[tid];
        out[tid].x = xv;
        out[tid].y = yv;
        out[tid].z = xv * xv + yv * yv;
        out[tid].w = __sqrtf(out[tid].z);
    }
}

// ------------------------------------------------------------------
// Struct passed by value to device function.

__device__ float vec4_dot(Vec4f a, Vec4f b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__global__ void vec4_dot_kernel(float *out, Vec4f *a, Vec4f *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = vec4_dot(a[tid], b[tid]);
    }
}
