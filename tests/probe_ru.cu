// Probe: out-parameter patterns, multi-value returns via pointers,
// nested struct access, array-of-structs vs struct-of-arrays,
// and complex indexed stores.

struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };
struct Particle { float px, py, pz, vx, vy, vz; };

// ------------------------------------------------------------------
// Out-parameter: device function writes two results via pointers.

__device__ void sincos_approx(float angle, float *s, float *c) {
    // Approximate sin/cos via Taylor (just arithmetic, no libm call)
    float a2 = angle * angle;
    *s = angle * (1.0f - a2 / 6.0f);
    *c = 1.0f - a2 / 2.0f;
}

__global__ void sincos_kernel(float *sout, float *cout, float *angles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s, c;
        sincos_approx(angles[tid], &s, &c);
        sout[tid] = s;
        cout[tid] = c;
    }
}

// ------------------------------------------------------------------
// Struct-of-arrays (SoA) access: separate arrays for each field.

__global__ void soa_update(float *px, float *py, float *pz,
                            float *vx, float *vy, float *vz,
                            float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        px[tid] += vx[tid] * dt;
        py[tid] += vy[tid] * dt;
        pz[tid] += vz[tid] * dt;
    }
}

// ------------------------------------------------------------------
// Array-of-structs (AoS): struct pointer arithmetic.

__global__ void aos_update(Particle *parts, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        parts[tid].px += parts[tid].vx * dt;
        parts[tid].py += parts[tid].vy * dt;
        parts[tid].pz += parts[tid].vz * dt;
    }
}

// ------------------------------------------------------------------
// Nested struct: Vec2 inside Vec3-like layout.

__global__ void vec_dot(float *out, Vec3 *a, Vec3 *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float d = a[tid].x * b[tid].x
                + a[tid].y * b[tid].y
                + a[tid].z * b[tid].z;
        out[tid] = d;
    }
}

// ------------------------------------------------------------------
// Write to indexed output with offset: multiple stores in one thread.

__global__ void scatter_write(float *out, float *in, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid];
        out[i * 3 + 0] = in[tid];
        out[i * 3 + 1] = in[tid] * 2.0f;
        out[i * 3 + 2] = in[tid] * 3.0f;
    }
}

// ------------------------------------------------------------------
// Reduction via shared memory + sequential addressing.

__global__ void reduce_sum(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

// ------------------------------------------------------------------
// Conditional scatter: only write if predicate is true.

__global__ void cond_scatter(int *out, int *in, int *mask, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (mask[tid]) {
            out[tid] = in[tid] * 2;
        }
    }
}
