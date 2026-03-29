// Probe: Multiple kernels accessing shared structs, complex function signatures
// - Function taking a struct by pointer and returning void
// - Recursive struct composition (struct A has struct B field)
// - Multi-return struct pattern (output via pointer params)
// - Pointer to struct array passed to device func

struct Complex {
    float re, im;
};

__device__ Complex complex_mul(Complex a, Complex b) {
    Complex r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

__device__ float complex_abs(Complex c) {
    return sqrtf(c.re * c.re + c.im * c.im);
}

__global__ void mandelbrot(int *out, float x0, float y0, float dx, float dy,
                           int W, int H, int max_iter) {
    int px = threadIdx.x + blockIdx.x * blockDim.x;
    int py = threadIdx.y + blockIdx.y * blockDim.y;
    if (px >= W || py >= H) return;

    Complex c;
    c.re = x0 + px * dx;
    c.im = y0 + py * dy;

    Complex z;
    z.re = 0.0f;
    z.im = 0.0f;

    int iter = 0;
    while (iter < max_iter && complex_abs(z) < 2.0f) {
        z = complex_mul(z, z);
        z.re += c.re;
        z.im += c.im;
        iter++;
    }
    out[py * W + px] = iter;
}
