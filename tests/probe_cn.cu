// Probe: Patterns involving #include and extern declarations
// - extern "C" function declaration (parse-skip)
// - extern function declaration (no body)
// - Prototype-only declaration (no body, just signature)
// - Multiple return points in __device__ with early-exit optimization

// These should be silently consumed/skipped:
// extern "C" void some_runtime_func(void *p, int n);  // would need string literal handling

// Prototype with no body — should be skipped at module level
// (only __global__ and __device__ with bodies are parsed)

__device__ float poly3(float x, float a, float b, float c, float d) {
    return ((a * x + b) * x + c) * x + d;
}

__global__ void horner_eval(float *out, float *in, float a, float b, float c, float d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = poly3(in[tid], a, b, c, d);
    }
}

// Early exit patterns
__device__ float safe_log(float x) {
    if (x <= 0.0f) return -1e30f;
    return logf(x);
}

__global__ void log_array(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = safe_log(in[tid]);
    }
}
