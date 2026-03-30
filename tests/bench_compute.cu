// Compute-bound benchmarks: ALU-heavy kernels where instruction
// scheduling and register allocation actually matter.

// Polynomial evaluation (high arithmetic intensity)
__global__ void bench_poly(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = a[gid];
        // Horner's method, degree 15 — 15 FMA ops per element
        float r = 0.01f;
        r = r * x + 0.02f;
        r = r * x + 0.03f;
        r = r * x + 0.04f;
        r = r * x + 0.05f;
        r = r * x + 0.06f;
        r = r * x + 0.07f;
        r = r * x + 0.08f;
        r = r * x + 0.09f;
        r = r * x + 0.10f;
        r = r * x + 0.11f;
        r = r * x + 0.12f;
        r = r * x + 0.13f;
        r = r * x + 0.14f;
        r = r * x + 0.15f;
        out[gid] = r;
    }
}

// N-body style: each thread computes interaction with 64 "particles"
__global__ void bench_nbody(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float ax = 0.0f, ay = 0.0f;
        float px = a[gid], py = b[gid];
        for (int j = 0; j < 64; j++) {
            float dx = a[j % n] - px;
            float dy = b[j % n] - py;
            float r2 = dx*dx + dy*dy + 0.001f;
            float inv = rsqrtf(r2);
            float inv3 = inv * inv * inv;
            ax += dx * inv3;
            ay += dy * inv3;
        }
        out[gid] = ax + ay;
    }
}

// Mandelbrot: pure compute, divergent iteration counts
__global__ void bench_mandelbrot(int *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float cr = a[gid] * 3.0f - 2.0f;  // map to [-2, 1]
        float ci = b[gid] * 2.0f - 1.0f;  // map to [-1, 1]
        float zr = 0.0f, zi = 0.0f;
        int iter = 0;
        while (zr*zr + zi*zi < 4.0f && iter < 256) {
            float tmp = zr*zr - zi*zi + cr;
            zi = 2.0f*zr*zi + ci;
            zr = tmp;
            iter++;
        }
        out[gid] = iter;
    }
}
