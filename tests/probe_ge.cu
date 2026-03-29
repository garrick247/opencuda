// Probe: global variable with complex initializer, multiple translation
// unit style (__device__ functions as forward declarations),
// const global array used across multiple kernels

__constant__ float PI = 3.14159265358979f;
__constant__ float TWO_PI = 6.28318530717959f;
__constant__ float HALF_PI = 1.57079632679490f;

__device__ float fast_sin(float x) {
    // Normalize to [-pi, pi]
    while (x > PI) x -= TWO_PI;
    while (x < -PI) x += TWO_PI;
    // Bhaskara approximation
    float x2 = x * x;
    return x * (PI * PI - x2) / (HALF_PI * PI * PI - x2 * 0.25f);
}

__device__ float fast_cos(float x) {
    return fast_sin(x + HALF_PI);
}

__global__ void sincos_kernel(float *out_s, float *out_c, float *angles, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = angles[tid];
        out_s[tid] = fast_sin(a);
        out_c[tid] = fast_cos(a);
    }
}

// Use PI in computation
__global__ void angle_normalize(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];
        // Normalize to [0, 2*pi)
        while (a < 0.0f) a += TWO_PI;
        while (a >= TWO_PI) a -= TWO_PI;
        out[tid] = a / TWO_PI;  // map to [0, 1)
    }
}
