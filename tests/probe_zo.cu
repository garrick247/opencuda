// Probe: scientific computing — Runge-Kutta 4th order ODE step,
// FFT butterfly (radix-2 Cooley-Tukey), tridiagonal solver (Thomas),
// Monte Carlo pi estimation, Jacobi iteration, and conjugate gradient step.

// ------------------------------------------------------------------
// RK4 single step for dy/dt = f(t, y) = -y (exponential decay).

__device__ float rk4_f(float t, float y) { return -y; }

__global__ void rk4_step(float *y_out, float *y_in, float *t_in,
                            float dt, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float t = t_in[gid], y = y_in[gid];
    float k1 = rk4_f(t, y);
    float k2 = rk4_f(t + dt/2, y + dt/2*k1);
    float k3 = rk4_f(t + dt/2, y + dt/2*k2);
    float k4 = rk4_f(t + dt, y + dt*k3);
    y_out[gid] = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
}

// ------------------------------------------------------------------
// FFT butterfly: complex multiply-add for radix-2 Cooley-Tukey.
// (real, imag) stored interleaved.

__global__ void fft_butterfly(float *data_r, float *data_i,
                                 float *tw_r, float *tw_i,
                                 int half_size, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n / 2) return;
    int group = gid / half_size;
    int pair  = gid % half_size;
    int i = group * 2 * half_size + pair;
    int j = i + half_size;
    // Twiddle factor
    float wr = tw_r[pair];
    float wi = tw_i[pair];
    // Butterfly
    float tr = data_r[j] * wr - data_i[j] * wi;
    float ti = data_r[j] * wi + data_i[j] * wr;
    data_r[j] = data_r[i] - tr;
    data_i[j] = data_i[i] - ti;
    data_r[i] = data_r[i] + tr;
    data_i[i] = data_i[i] + ti;
}

// ------------------------------------------------------------------
// Jacobi iteration for Laplace equation (2D grid).

__global__ void jacobi_step(float *next, float *curr, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= W-1 || y < 1 || y >= H-1) return;
    next[y*W+x] = 0.25f * (curr[(y-1)*W+x] + curr[(y+1)*W+x]
                          + curr[y*W+(x-1)] + curr[y*W+(x+1)]);
}

// ------------------------------------------------------------------
// Monte Carlo pi estimation per-thread.

__global__ void monte_carlo_pi(int *hits, unsigned seed, int samples_per_thread, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned state = seed + (unsigned)gid * 2654435761u;
    int count = 0;
    for (int i = 0; i < samples_per_thread; i++) {
        state = state * 1664525u + 1013904223u;
        float x = (float)(state & 0xFFFF) / 65535.0f;
        state = state * 1664525u + 1013904223u;
        float y = (float)(state & 0xFFFF) / 65535.0f;
        if (x*x + y*y <= 1.0f) count++;
    }
    atomicAdd(hits, count);
}

// ------------------------------------------------------------------
// Conjugate gradient: r = b - A*x, dot(r,r), p = r.

__global__ void cg_init(float *r, float *p, float *b, float *Ax, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float ri = b[gid] - Ax[gid];
        r[gid] = ri;
        p[gid] = ri;
    }
}

// x = x + alpha*p
__global__ void cg_update_x(float *x, float *p, float alpha, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) x[gid] += alpha * p[gid];
}

// r = r - alpha*Ap
__global__ void cg_update_r(float *r, float *Ap, float alpha, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) r[gid] -= alpha * Ap[gid];
}

// p = r + beta*p
__global__ void cg_update_p(float *p, float *r, float beta, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) p[gid] = r[gid] + beta * p[gid];
}
