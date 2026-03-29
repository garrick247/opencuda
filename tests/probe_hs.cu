// Probe: __device__ functions with multiple output paths via output params,
// recursive-style depth-limited iterative descent,
// returning struct by value (via decomposed fields)

struct Float2 {
    float x;
    float y;
};

__device__ Float2 make_float2(float x, float y) {
    Float2 r;
    r.x = x;
    r.y = y;
    return r;
}

__device__ Float2 complex_mul(Float2 a, Float2 b) {
    Float2 r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}

__global__ void mandelbrot_iter(int *out, float *cx, float *cy, int max_iter, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Float2 c = make_float2(cx[tid], cy[tid]);
        Float2 z = make_float2(0.0f, 0.0f);
        int iter = 0;
        while (iter < max_iter && z.x * z.x + z.y * z.y < 4.0f) {
            z = complex_mul(z, z);
            z.x += c.x;
            z.y += c.y;
            iter++;
        }
        out[tid] = iter;
    }
}
