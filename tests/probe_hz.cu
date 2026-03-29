// Probe: function pointer-style dispatch via switch on type tag,
// multiple kernels sharing a device helper,
// array of structs vs struct of arrays,
// nested ternary

struct Particle {
    float x, y, vx, vy;
};

__device__ float dist2(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return dx*dx + dy*dy;
}

// Array of structs access
__global__ void integrate(Particle *p, float dt, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        p[tid].x += p[tid].vx * dt;
        p[tid].y += p[tid].vy * dt;
    }
}

// Nested ternary: clamp(x, lo, hi)
__global__ void clamp_kernel(float *out, float *in, float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        out[tid] = (v < lo) ? lo : (v > hi) ? hi : v;
    }
}

// Distance computation using device function
__global__ void pairwise_dist(float *out, Particle *p, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float d = 0.0f;
        for (int j = 0; j < n; j++) {
            if (j != tid) {
                d += dist2(p[tid].x, p[tid].y, p[j].x, p[j].y);
            }
        }
        out[tid] = d;
    }
}
