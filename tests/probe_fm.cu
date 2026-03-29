// Probe: complex preprocessor — nested macro expansion, macro that expands
// to a full expression, stringification ignored, concatenation

#define ALPHA 0.1f
#define BETA  0.9f
#define EPS   1e-8f

#define LERP(a, b, t)  ((a) * (1.0f - (t)) + (b) * (t))
#define NORM_SQ(x, y)  ((x) * (x) + (y) * (y))
#define DIST(x1, y1, x2, y2)  sqrtf(NORM_SQ((x2)-(x1), (y2)-(y1)))
#define SAFE_DIV(a, b) ((b) == 0.0f ? 0.0f : (a) / (b))

#define IDX2D(row, col, stride) ((row) * (stride) + (col))

__global__ void macro_expand(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float lerped = LERP(ALPHA, BETA, v);
        float ns = NORM_SQ(v, lerped);
        out[tid] = SAFE_DIV(ns, ns + EPS);
    }
}

__global__ void macro_2d(float *out, float *in, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = IDX2D(row, col, cols);
        float d = DIST((float)row, (float)col, 0.0f, 0.0f);
        out[idx] = in[idx] * d;
    }
}
