// Probe: texture-like LUT pattern (constant memory + cached access),
// multiple __constant__ arrays with different types,
// __constant__ struct array

struct Coeff {
    float a, b, c;
};

__constant__ Coeff g_coeffs[8];
__constant__ int g_lut[256];
__constant__ float g_gauss[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};

__global__ void poly_eval(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int coeff_idx = tid & 7;
        float x = in[tid];
        Coeff c = g_coeffs[coeff_idx];
        out[tid] = c.a * x * x + c.b * x + c.c;
    }
}

__global__ void lut_remap(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] & 0xFF;
        out[tid] = g_lut[v];
    }
}

__global__ void gauss_blur(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid >= 2 && tid < n - 2) {
        float sum = 0.0f;
        for (int k = 0; k < 5; k++) {
            sum += g_gauss[k] * in[tid - 2 + k];
        }
        out[tid] = sum;
    }
}
