// Probe: complex inline function with multiple early returns and
// a struct as return type — this exercises the return-merge phi chain

struct Complex {
    float re, im;
};

__device__ Complex complex_mul(Complex a, Complex b) {
    Complex r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

__device__ Complex complex_add(Complex a, Complex b) {
    Complex r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

__device__ float complex_abs_sq(Complex a) {
    return a.re * a.re + a.im * a.im;
}

// Mandelbrot set membership (limited iterations)
__device__ int mandelbrot(float cx, float cy, int max_iter) {
    Complex z;
    z.re = 0.0f;
    z.im = 0.0f;
    Complex c;
    c.re = cx;
    c.im = cy;
    for (int i = 0; i < max_iter; i++) {
        z = complex_add(complex_mul(z, z), c);
        if (complex_abs_sq(z) > 4.0f) return i;
    }
    return max_iter;
}

__global__ void mandelbrot_kernel(int *out, float x0, float y0,
                                   float dx, float dy, int w, int h,
                                   int max_iter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        float cx = x0 + col * dx;
        float cy = y0 + row * dy;
        out[row * w + col] = mandelbrot(cx, cy, max_iter);
    }
}
