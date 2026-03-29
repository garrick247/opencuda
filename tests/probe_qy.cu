// Probe: __device__ fn called with global struct array element by value,
// global struct read-modify-write in loop, const pointer params,
// multi-block grid patterns, and int64 struct fields.

// ------------------------------------------------------------------
// __device__ fn called with global struct array element.

struct Sensor {
    float temp;
    float pressure;
    int   id;
};

__device__ Sensor g_sensors[16];

__device__ float sensor_index(Sensor s) {
    // index = temp / pressure
    return (s.pressure > 0.0f) ? s.temp / s.pressure : 0.0f;
}

__global__ void compute_indices(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        // Pass global struct array element by value to device fn
        out[tid] = sensor_index(g_sensors[tid]);
    }
}

// ------------------------------------------------------------------
// Global struct array read-modify-write in a loop.

struct Node2 {
    float val;
    int   active;
};

__device__ Node2 g_nodes[8];

__global__ void relax_nodes(float alpha, int iters) {
    if (threadIdx.x == 0) {
        for (int it = 0; it < iters; it++) {
            for (int i = 1; i < 7; i++) {
                if (g_nodes[i].active) {
                    g_nodes[i].val = g_nodes[i].val * (1.0f - alpha)
                                   + (g_nodes[i-1].val + g_nodes[i+1].val)
                                   * alpha * 0.5f;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// const pointer parameter: read-only input.

__global__ void const_ptr(float *out, const float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2.0f + 1.0f;
    }
}

// ------------------------------------------------------------------
// Multi-block grid with blockIdx / gridDim arithmetic.

__global__ void block_sum(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // Each block writes its partial sum to out[blockIdx.x]
        float val = in[gid];
        // Simple: just store each element (caller reduces)
        out[gid] = val * (float)(blockIdx.x + 1);
    }
}

// ------------------------------------------------------------------
// Struct with int64 field.

struct Event {
    long long timestamp;
    int       type;
    float     value;
};

__device__ Event g_events[4];

__global__ void record_event(long long ts, int type, float val, int slot) {
    if (threadIdx.x == 0 && slot < 4) {
        g_events[slot].timestamp = ts;
        g_events[slot].type      = type;
        g_events[slot].value     = val;
    }
}

__global__ void read_events(long long *ts_out, int *type_out,
                             float *val_out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 4) {
        ts_out[tid]   = g_events[tid].timestamp;
        type_out[tid] = g_events[tid].type;
        val_out[tid]  = g_events[tid].value;
    }
}

// ------------------------------------------------------------------
// __device__ fn with loop that reads a const pointer.

__device__ float sum_const(const float *p, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += p[i];
    }
    return s;
}

__global__ void sum_windows(float *out, const float *in, int win, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = sum_const(in + tid, win);
    }
}
