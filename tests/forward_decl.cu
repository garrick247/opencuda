// Regression: __device__ function forward declarations (prototypes)
// Without fix: _parse_device_func expected LBRACE after param list, got SEMI →
//   ParseError "expected LBRACE, got SEMI ';'".
// Fix: _parse_device_func checks for SEMI after RPAREN; if found, returns early
//   (prototype with no body — definition will follow).

// Forward declarations
__device__ float dot2(float ax, float ay, float bx, float by);
__device__ int imax(int a, int b);
__device__ float sigmoid(float x);

// Kernel using forward-declared functions
__global__ void fwd_decl_test(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = data[tid * 2 + 0];
        float b = data[tid * 2 + 1];
        float d = dot2(a, b, a, b);
        float s = sigmoid(d);
        out[tid] = s;
    }
}

// Definitions following the forward declarations
__device__ float dot2(float ax, float ay, float bx, float by) {
    return ax * bx + ay * by;
}

__device__ int imax(int a, int b) {
    return a > b ? a : b;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
