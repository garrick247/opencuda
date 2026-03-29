// Probe: mixed signed/unsigned comparisons, implicit int promotion in shifts,
// 64-bit shift operations, negative array index (pointer behind base),
// __device__ function with default-parameter-like multiple overloads (via macros),
// struct field as lvalue in augmented assignment, global memory fence,
// atomicAdd on float*, and __threadfence/__threadfence_block/__threadfence_system.

// ------------------------------------------------------------------
// Mixed signed/unsigned comparisons.

__global__ void mixed_sign_cmp(int *out, int *si, unsigned *ui, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   s = si[tid];
        unsigned u = ui[tid];
        // Explicit cast to avoid UB
        int r = ((unsigned)s < u) ? 1 :
                ((unsigned)s > u) ? -1 : 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// 64-bit shift operations.

__global__ void shift64(long long *out_l, unsigned long long *out_r,
                          long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long v = in[tid];
        out_l[tid] = v << 3;          // 64-bit left shift
        out_r[tid] = (unsigned long long)v >> 2;  // logical right shift
    }
}

// ------------------------------------------------------------------
// Struct field as lvalue in augmented assignment.

struct Counter { int val; int steps; };

__device__ void counter_step(struct Counter *c, int delta) {
    c->val   += delta;
    c->steps += 1;
}

__global__ void counter_kernel(int *out_val, int *out_steps,
                                  int *deltas, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Counter c;
        c.val   = 0;
        c.steps = 0;
        for (int i = 0; i < 4; i++) {
            counter_step(&c, deltas[(tid + i) % n]);
        }
        out_val[tid]   = c.val;
        out_steps[tid] = c.steps;
    }
}

// ------------------------------------------------------------------
// __threadfence / __threadfence_block / __threadfence_system.

__global__ void threadfence_test(int *flag, int *data, int val, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        data[tid] = val;
        __threadfence();
        flag[tid] = 1;
    }
}

__global__ void threadfence_block_test(int *shared_flag, int *out, int n) {
    __shared__ int s[256];
    int tid = threadIdx.x;
    s[tid] = tid;
    __threadfence_block();
    out[tid] = s[(tid + 1) % blockDim.x];
}

// ------------------------------------------------------------------
// atomicAdd on float*.

__global__ void atomic_float_add(float *sum, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) atomicAdd(sum, in[gid]);
}

// ------------------------------------------------------------------
// Implicit int promotion in shifts.

__global__ void shift_promotion(int *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char b = in[tid];
        // b is promoted to int before shift
        int r = (int)b << 8 | (int)b;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Macro that simulates overloaded default parameter.

#define CLAMP_INT(v)       ((v) < 0 ? 0 : (v) > 255 ? 255 : (v))
#define CLAMP_RANGE(v,lo,hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))

__global__ void macro_overload(int *out_a, int *out_b,
                                  int *in, int lo, int hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out_a[tid] = CLAMP_INT(in[tid]);
        out_b[tid] = CLAMP_RANGE(in[tid], lo, hi);
    }
}

// ------------------------------------------------------------------
// Global memory fence pattern: producer-consumer flag.

__global__ void produce(int *data, int *flag, int val, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        data[tid] = val + tid;
        __threadfence();
        atomicExch(&flag[tid], 1);
    }
}

// ------------------------------------------------------------------
// Complex pointer aliasing: two pointers into same array, different offsets.

__global__ void dual_ptr_sum(int *out, int *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p1 = in + tid;
        int *p2 = in + (tid + stride) % n;
        out[tid] = *p1 + *p2;
    }
}
