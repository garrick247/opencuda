// Probe: address-of local struct fields, local struct arrays with
// pointer access, multiple passes over same global struct, and
// complex struct-through-pointer write patterns.

// ------------------------------------------------------------------
// Address of local struct field: &local.field passed to atomicAdd.

struct Counter2 {
    int hits;
    int misses;
};

__global__ void local_field_addr(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Counter2 c;
        c.hits = 0;
        c.misses = 0;
        for (int i = 0; i < n; i++) {
            int *p = (data[i] > 0) ? &c.hits : &c.misses;
            (*p)++;
        }
        out[0] = c.hits;
        out[1] = c.misses;
    }
}

// ------------------------------------------------------------------
// Local struct array: access via pointer.

struct PairF {
    float a, b;
};

__global__ void local_struct_array(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        PairF pairs[4];
        for (int i = 0; i < 4; i++) {
            pairs[i].a = in[tid * 8 + i * 2 + 0];
            pairs[i].b = in[tid * 8 + i * 2 + 1];
        }
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum += pairs[i].a * pairs[i].b;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// __device__ fn takes pointer-to-struct and returns modified copy.

struct Vec3f {
    float x, y, z;
};

__device__ Vec3f normalize(Vec3f *v) {
    float len = v->x*v->x + v->y*v->y + v->z*v->z;
    Vec3f r;
    if (len > 0.0f) {
        float inv = 1.0f / len;
        r.x = v->x * inv;
        r.y = v->y * inv;
        r.z = v->z * inv;
    } else {
        r.x = 0.0f; r.y = 0.0f; r.z = 0.0f;
    }
    return r;
}

__global__ void normalize_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec3f v;
        v.x = in[tid * 3 + 0];
        v.y = in[tid * 3 + 1];
        v.z = in[tid * 3 + 2];
        Vec3f r = normalize(&v);
        out[tid * 3 + 0] = r.x;
        out[tid * 3 + 1] = r.y;
        out[tid * 3 + 2] = r.z;
    }
}

// ------------------------------------------------------------------
// Multiple passes over same global struct: write then read in sequence.

struct Cache {
    float val;
    int   valid;
};

__device__ Cache g_cache[8];

__global__ void fill_cache(float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        g_cache[tid].val   = data[tid];
        g_cache[tid].valid = 1;
    }
}

__global__ void evict_stale(float threshold) {
    int tid = threadIdx.x;
    if (tid < 8) {
        if (g_cache[tid].valid && g_cache[tid].val < threshold) {
            g_cache[tid].valid = 0;
            g_cache[tid].val   = 0.0f;
        }
    }
}

__global__ void read_cache(float *out, int *valid_out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        out[tid]       = g_cache[tid].val;
        valid_out[tid] = g_cache[tid].valid;
    }
}

// ------------------------------------------------------------------
// Struct written through computed pointer in loop.

__device__ Vec3f g_trail[16];

__global__ void record_trail(float *path, int n) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < n && i < 16; i++) {
            Vec3f *p = &g_trail[i];
            p->x = path[i * 3 + 0];
            p->y = path[i * 3 + 1];
            p->z = path[i * 3 + 2];
        }
    }
}

__global__ void trail_lengths(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        Vec3f *p = &g_trail[tid];
        out[tid] = p->x*p->x + p->y*p->y + p->z*p->z;
    }
}
