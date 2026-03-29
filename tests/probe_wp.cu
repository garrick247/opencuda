// Probe: complex struct passing patterns — struct with mixed types,
// device fn returning struct with pointer, inline expansion with struct
// array parameter, and high register pressure scenarios.

// ------------------------------------------------------------------
// Struct with 4 field types: int, float, double, pointer.

struct Mixed {
    int    count;
    float  weight;
    double value;
    int   *ref;
};

__global__ void mixed_struct_fields(int *out, int *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Mixed m;
        m.count  = tid;
        m.weight = (float)tid * 0.5f;
        m.value  = (double)tid * 1.5;
        m.ref    = buf + tid;

        *m.ref = m.count + (int)m.weight + (int)m.value;
        out[tid] = *m.ref;  // tid + tid/2 + tid*3/2 (integer arithmetic)
    }
}

// ------------------------------------------------------------------
// Device fn returning a struct with multiple computations.

struct Result2 { int lo; int hi; };

__device__ struct Result2 split_value(int v) {
    struct Result2 r;
    r.lo = v & 0xFF;
    r.hi = (v >> 8) & 0xFF;
    return r;
}

__global__ void struct_return_multi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Result2 r = split_value(in[tid]);
        out[tid] = r.lo + r.hi;
    }
}

// ------------------------------------------------------------------
// Struct passed to device fn, struct returned.

struct Transform { float scale; float offset; };

__device__ float apply_transform(float v, struct Transform t) {
    return v * t.scale + t.offset;
}

__global__ void struct_param_and_use(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Transform t;
        t.scale  = 2.0f;
        t.offset = 1.0f;
        out[tid] = apply_transform(in[tid], t);
    }
}

// ------------------------------------------------------------------
// Multiple struct instances, one modified, one not.

struct Pair { int first; int second; };

__global__ void struct_selective_modify(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Pair a;
        a.first  = in[tid];
        a.second = in[tid] * 2;

        struct Pair b = a;   // copy
        b.first  += 10;      // only modify b

        out[tid] = a.first + b.first;  // in[tid] + (in[tid]+10)
    }
}

// ------------------------------------------------------------------
// High register pressure: 20 independent computations.

__global__ void high_reg_pressure(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r0  = v + 0;
        int r1  = v + 1;
        int r2  = v + 2;
        int r3  = v + 3;
        int r4  = v + 4;
        int r5  = v + 5;
        int r6  = v + 6;
        int r7  = v + 7;
        int r8  = v + 8;
        int r9  = v + 9;
        int r10 = v + 10;
        int r11 = v + 11;
        int r12 = v + 12;
        int r13 = v + 13;
        int r14 = v + 14;
        int r15 = v + 15;
        int r16 = v + 16;
        int r17 = v + 17;
        int r18 = v + 18;
        int r19 = v + 19;
        // Sum: 20v + (0+1+...+19) = 20v + 190
        int sum = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9
                + r10+r11+r12+r13+r14+r15+r16+r17+r18+r19;
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Struct array parameter to kernel.

struct Particle { float x; float y; float vx; float vy; };

__global__ void update_particles(struct Particle *particles, float dt, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        particles[tid].x += particles[tid].vx * dt;
        particles[tid].y += particles[tid].vy * dt;
    }
}

// ------------------------------------------------------------------
// __device__ fn modifying struct through pointer, then using result.

__device__ void normalize_pair(struct Pair *p, int total) {
    if (total > 0) {
        p->first  = (p->first  * 100) / total;
        p->second = (p->second * 100) / total;
    }
}

__global__ void normalize_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Pair p;
        p.first  = in[tid];
        p.second = in[tid] * 2;
        int total = p.first + p.second;   // 3*in[tid]
        normalize_pair(&p, total);
        out[tid] = p.first + p.second;    // ≈ 33 + 66 = 99 (integer div truncation)
    }
}
