// Probe: global struct arrays — field access, write, read, array of struct
// with multiple fields, and struct-of-arrays vs array-of-structs patterns.
// Exercises the v0.88 (struct size) and v0.89 (array subscript + field) fixes.

// ------------------------------------------------------------------
// Array of structs: write and read back.

struct Particle {
    float x, y;
    float vx, vy;
};

__device__ Particle g_particles[32];

__global__ void update_particles(float dt) {
    int tid = threadIdx.x;
    if (tid < 32) {
        g_particles[tid].x += g_particles[tid].vx * dt;
        g_particles[tid].y += g_particles[tid].vy * dt;
    }
}

__global__ void read_particles(float *out_x, float *out_y) {
    int tid = threadIdx.x;
    if (tid < 32) {
        out_x[tid] = g_particles[tid].x;
        out_y[tid] = g_particles[tid].y;
    }
}

// ------------------------------------------------------------------
// Global struct with mixed int/float: write from kernel input.

struct KeyValue {
    int key;
    float val;
};

__device__ KeyValue g_kv_table[16];

__global__ void fill_kv(int *keys, float *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        g_kv_table[tid].key = keys[tid];
        g_kv_table[tid].val = vals[tid];
    }
}

__global__ void lookup_kv(float *out, int *query_keys, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int q = 0; q < n; q++) {
            int qk = query_keys[q];
            float found = -1.0f;
            for (int i = 0; i < 16; i++) {
                if (g_kv_table[i].key == qk) {
                    found = g_kv_table[i].val;
                    break;
                }
            }
            out[q] = found;
        }
    }
}

// ------------------------------------------------------------------
// Large struct array: 3D vectors.

struct Vec3 {
    float x, y, z;
};

__device__ Vec3 g_vecs[8];

__global__ void compute_dot_products(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        float dx = g_vecs[tid].x;
        float dy = g_vecs[tid].y;
        float dz = g_vecs[tid].z;
        out[tid] = dx * dx + dy * dy + dz * dz;
    }
}

// ------------------------------------------------------------------
// Global struct with ptr field (if supported) — skip for now.
// Instead: global struct with two ints.

struct Stats2 {
    int count;
    int total;
};

__device__ Stats2 g_stats2[4];

__global__ void accumulate_stats(int *data, int n, int bucket_count) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int bucket = v % bucket_count;
        if (bucket < 4) {
            atomicAdd(&g_stats2[bucket].count, 1);
            atomicAdd(&g_stats2[bucket].total, v);
        }
    }
}

__global__ void read_stats(int *out_count, int *out_total) {
    int tid = threadIdx.x;
    if (tid < 4) {
        out_count[tid] = g_stats2[tid].count;
        out_total[tid] = g_stats2[tid].total;
    }
}
