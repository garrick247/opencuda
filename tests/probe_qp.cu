// Probe: passing &global_struct_arr[i] to __device__ functions,
// global struct field increment (g_s.field++), loop-carried struct
// fields via PhiInst, and mixed-type struct field coercions.

// ------------------------------------------------------------------
// Global struct array: pass element address to __device__ function.

struct Particle {
    float x, y;
    float vx, vy;
    float mass;
};

__device__ Particle g_parts[16];

__device__ void integrate(Particle *p, float dt) {
    p->x += p->vx * dt;
    p->y += p->vy * dt;
}

__device__ float kinetic_energy(Particle *p) {
    float v2 = p->vx * p->vx + p->vy * p->vy;
    return 0.5f * p->mass * v2;
}

__global__ void simulate(float dt, float *ke_out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        integrate(&g_parts[tid], dt);
        ke_out[tid] = kinetic_energy(&g_parts[tid]);
    }
}

// ------------------------------------------------------------------
// Global struct field increment (++ on global).

struct GlobalCounter {
    int n_events;
    int n_errors;
};

__device__ GlobalCounter g_counters;

__global__ void record_events(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            if (data[i] >= 0) {
                g_counters.n_events++;
            } else {
                g_counters.n_errors++;
            }
        }
    }
}

__global__ void read_counters(int *out_ev, int *out_err) {
    if (threadIdx.x == 0) {
        out_ev[0]  = g_counters.n_events;
        out_err[0] = g_counters.n_errors;
    }
}

// ------------------------------------------------------------------
// Loop-carried struct field: accumulator struct updated each iteration.

struct RunStats {
    float min_val;
    float max_val;
    float total;
    int   count;
};

__global__ void run_stats_kernel(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        RunStats s;
        s.min_val = in[0];
        s.max_val = in[0];
        s.total   = 0.0f;
        s.count   = 0;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v < s.min_val) s.min_val = v;
            if (v > s.max_val) s.max_val = v;
            s.total += v;
            s.count++;
        }
        out[0] = s.min_val;
        out[1] = s.max_val;
        out[2] = s.total;
        out[3] = (float)s.count;
    }
}

// ------------------------------------------------------------------
// Mixed-type struct: float and int fields used together.

struct SplitKey {
    unsigned int hi;
    unsigned int lo;
};

__device__ SplitKey split64(unsigned long long v) {
    SplitKey k;
    k.hi = (unsigned int)(v >> 32);
    k.lo = (unsigned int)(v & 0xFFFFFFFFULL);
    return k;
}

__global__ void split_keys(unsigned int *out_hi, unsigned int *out_lo,
                            unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        SplitKey k = split64(in[tid]);
        out_hi[tid] = k.hi;
        out_lo[tid] = k.lo;
    }
}

// ------------------------------------------------------------------
// Global struct field compound assignment.

struct Accum2 {
    float pos_sum;
    float neg_sum;
};

__device__ Accum2 g_accum2;

__global__ void accum_pos_neg(float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v >= 0.0f) {
                g_accum2.pos_sum += v;
            } else {
                g_accum2.neg_sum += v;
            }
        }
    }
}
