// Probe: push the edges — very large struct (8+ fields), deeply nested
// device function calls (5 levels, mix of inlined + recursive), multiple
// __shared__ arrays in one kernel, complex pointer aliasing through
// device function, warp divergence with nested ternary + ballot,
// and kernel that uses every CUDA builtin we support.

// ------------------------------------------------------------------
// Large struct with 10 fields.

struct Particle {
    float px, py, pz;
    float vx, vy, vz;
    float mass;
    float charge;
    int   type;
    int   active;
};

__device__ float kinetic_energy(struct Particle p) {
    float v2 = p.vx*p.vx + p.vy*p.vy + p.vz*p.vz;
    return 0.5f * p.mass * v2;
}

__global__ void particle_energy(float *out, float *px, float *py, float *pz,
                                   float *vx, float *vy, float *vz,
                                   float *mass, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Particle p;
        p.px = px[tid]; p.py = py[tid]; p.pz = pz[tid];
        p.vx = vx[tid]; p.vy = vy[tid]; p.vz = vz[tid];
        p.mass = mass[tid]; p.charge = 0.0f;
        p.type = 0; p.active = 1;
        out[tid] = kinetic_energy(p);
    }
}

// ------------------------------------------------------------------
// 5-level inlined call chain.

__device__ int l5(int x) { return x + 1; }
__device__ int l4(int x) { return l5(x) * 2; }
__device__ int l3(int x) { return l4(x) + l4(x - 1); }
__device__ int l2(int x) { return l3(x) - l3(x / 2); }
__device__ int l1(int x) { return l2(x) + l2(x + 1); }

__global__ void deep_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = l1(in[tid]);
}

// ------------------------------------------------------------------
// Multiple __shared__ arrays in one kernel.

__global__ void multi_shared(float *out, float *in, int n) {
    __shared__ float buf_a[128];
    __shared__ float buf_b[128];
    __shared__ int   buf_idx[128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // Load into different shared arrays
    buf_a[tid]   = (gid < n) ? in[gid] : 0.0f;
    buf_b[tid]   = (gid < n) ? in[gid] * 2.0f : 0.0f;
    buf_idx[tid] = tid;
    __syncthreads();
    // Cross-reference
    int partner = buf_idx[127 - tid];
    float result = buf_a[tid] + buf_b[partner];
    if (gid < n) out[gid] = result;
}

// ------------------------------------------------------------------
// Kitchen-sink kernel: uses many builtins in one kernel.

__global__ void kitchen_sink(float *out, float *in, int n) {
    // Thread/block IDs
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;
    int lane = tid & 31;
    int wid  = tid >> 5;

    if (gid >= n) return;

    // Load with __ldg
    float v = __ldg(in + gid);

    // Math intrinsics
    float sv = __sinf(v);
    float cv = __cosf(v);
    float ev = __expf(-fabsf(v));

    // Warp vote
    int any_neg = __any_sync(0xFFFFFFFF, v < 0.0f);
    unsigned ballot = __ballot_sync(0xFFFFFFFF, v > 0.0f);
    int pop = __popc(ballot);

    // Warp shuffle: broadcast lane 0's value
    float broadcast = __shfl_sync(0xFFFFFFFF, v, 0);

    // Combine
    float result = sv + cv + ev + (float)any_neg + (float)pop + broadcast;

    // Shared memory reduction
    __shared__ float smem[256];
    smem[tid] = result;
    __syncthreads();
    if (tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64) smem[tid] += smem[tid + 64];
    __syncthreads();
    if (tid < 32) {
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid +  8];
        smem[tid] += smem[tid +  4];
        smem[tid] += smem[tid +  2];
        smem[tid] += smem[tid +  1];
    }

    // Atomic output
    if (tid == 0) atomicAdd(out, smem[0]);
}

// ------------------------------------------------------------------
// Complex ballot + popc pattern: count transitions in bitmask.

__global__ void count_transitions(int *out, int *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned v = (unsigned)in[gid];
    // Count bit transitions: XOR adjacent bits, popcount
    unsigned transitions = v ^ (v >> 1);
    out[gid] = __popc(transitions);
}

// ------------------------------------------------------------------
// Warp-cooperative: each warp finds its local min and broadcasts.

__global__ void warp_local_min(float *out, float *in, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float v = (gid < n) ? in[gid] : 1e30f;
    // Warp min reduce
    float t;
    t = __shfl_xor_sync(0xFFFFFFFF, v, 16); if (t < v) v = t;
    t = __shfl_xor_sync(0xFFFFFFFF, v,  8); if (t < v) v = t;
    t = __shfl_xor_sync(0xFFFFFFFF, v,  4); if (t < v) v = t;
    t = __shfl_xor_sync(0xFFFFFFFF, v,  2); if (t < v) v = t;
    t = __shfl_xor_sync(0xFFFFFFFF, v,  1); if (t < v) v = t;
    // Broadcast min to all lanes
    float warp_min = __shfl_sync(0xFFFFFFFF, v, 0);
    if (gid < n) out[gid] = warp_min;
}
