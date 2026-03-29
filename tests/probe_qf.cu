// Probe: global __device__ struct field writes — exercises the v0.82 fix
// where g_struct.field = val silently emitted a load instead of a store.

// ------------------------------------------------------------------
// Global struct with scalar fields: write and read back.

struct Config {
    int width;
    int height;
    float scale;
};

__device__ Config g_config;

__global__ void write_config(int w, int h, float s) {
    int tid = threadIdx.x;
    if (tid == 0) {
        g_config.width  = w;
        g_config.height = h;
        g_config.scale  = s;
    }
}

__global__ void read_config(int *out_w, int *out_h, float *out_s) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out_w[0] = g_config.width;
        out_h[0] = g_config.height;
        out_s[0] = g_config.scale;
    }
}

// ------------------------------------------------------------------
// Global struct with int + long long fields.

struct LargeConfig {
    int   version;
    long long timestamp;
    int   flags;
};

__device__ LargeConfig g_large;

__global__ void write_large(int ver, long long ts, int fl) {
    int tid = threadIdx.x;
    if (tid == 0) {
        g_large.version   = ver;
        g_large.timestamp = ts;
        g_large.flags     = fl;
    }
}

__global__ void read_large(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = g_large.version + g_large.flags;
    }
}

// ------------------------------------------------------------------
// Conditional write to global struct field.
// Tests that the store is emitted only on taken branch.

__device__ int g_max_val = 0;
__device__ int g_max_idx = -1;

struct MaxEntry {
    int val;
    int idx;
};

__device__ MaxEntry g_max_entry;

__global__ void conditional_struct_write(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            if (data[i] > g_max_entry.val) {
                g_max_entry.val = data[i];
                g_max_entry.idx = i;
            }
        }
    }
}

__global__ void read_max_entry(int *out_val, int *out_idx) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out_val[0] = g_max_entry.val;
        out_idx[0] = g_max_entry.idx;
    }
}

// ------------------------------------------------------------------
// Global struct: compound-assign to field.
// g_counter.count += 1

struct Counter {
    int count;
    int total;
};

__device__ Counter g_cnt;

__global__ void increment_struct(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        atomicAdd(&g_max_val, data[tid]);
    }
    if (tid == 0) {
        g_cnt.count += 1;
        g_cnt.total += n;
    }
}

__global__ void read_counter(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = g_cnt.count;
        out[1] = g_cnt.total;
    }
}
