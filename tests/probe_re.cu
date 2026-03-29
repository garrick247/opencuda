// Probe: many-param functions, typedef struct in complex contexts,
// enum in struct and switch, void __device__ with multiple out-ptrs,
// and passing scalar primitives by address.

// ------------------------------------------------------------------
// Function with many parameters (> 8): tests parameter passing.

__global__ void many_params(float *out, float a, float b, float c, float d,
                              float e, float f, float g, float h, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = a + b + c + d + e + f + g + h;
        out[tid] = sum * (float)(tid + 1);
    }
}

// ------------------------------------------------------------------
// typedef struct used in device fn, kernel, and global array.

typedef struct {
    float pos[3];
    float vel[3];
    float mass;
} Particle;

__device__ Particle g_ptcls[4];

__device__ void integrate_particle(Particle *p, float dt) {
    p->pos[0] += p->vel[0] * dt;
    p->pos[1] += p->vel[1] * dt;
    p->pos[2] += p->vel[2] * dt;
}

__global__ void simulate_particles(float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 4) {
        integrate_particle(&g_ptcls[tid], dt);
    }
}

// ------------------------------------------------------------------
// Enum in struct and switch.

typedef enum { IDLE = 0, RUNNING = 1, STOPPED = 2, ERROR = 3 } Status;

typedef struct {
    int    id;
    Status status;
    float  progress;
} Task;

__global__ void process_tasks(float *out, Task *tasks, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Task t = tasks[tid];
        float r;
        switch (t.status) {
            case IDLE:    r = 0.0f;         break;
            case RUNNING: r = t.progress;   break;
            case STOPPED: r = 1.0f;         break;
            default:      r = -1.0f;        break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// void __device__ fn writing to multiple output pointers.

__device__ void div_mod(int a, int b, int *quot, int *rem) {
    *quot = a / b;
    *rem  = a % b;
}

__global__ void divmod_kernel(int *out_q, int *out_r,
                               int *num, int *den, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int q, r;
        div_mod(num[tid], den[tid], &q, &r);
        out_q[tid] = q;
        out_r[tid] = r;
    }
}

// ------------------------------------------------------------------
// Passing scalar by address to __device__ fn that modifies it.

__device__ void clamp_inplace(float *v, float lo, float hi) {
    if (*v < lo) *v = lo;
    if (*v > hi) *v = hi;
}

__global__ void clamp_kernel(float *data, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        clamp_inplace(&data[tid], lo, hi);
    }
}

// ------------------------------------------------------------------
// typedef struct array with inline array field accessed via device fn.

typedef struct {
    float samples[4];
    float mean;
} Block4;

__device__ void compute_mean(Block4 *b) {
    float s = b->samples[0] + b->samples[1]
            + b->samples[2] + b->samples[3];
    b->mean = s * 0.25f;
}

__global__ void block_means(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Block4 b;
        b.samples[0] = in[tid * 4 + 0];
        b.samples[1] = in[tid * 4 + 1];
        b.samples[2] = in[tid * 4 + 2];
        b.samples[3] = in[tid * 4 + 3];
        compute_mean(&b);
        out[tid] = b.mean;
    }
}
