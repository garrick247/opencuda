// Probe: inline math functions — fminf, fmaxf, fabsf, sqrtf, rsqrtf,
// __expf, __logf, __sinf, __cosf, __powf

__global__ void math_builtins(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float a = fabsf(v);
        float s = sqrtf(a);
        float r = rsqrtf(a + 1.0f);
        float lo = fminf(s, r);
        float hi = fmaxf(s, r);
        out[tid] = lo + hi;
    }
}

__global__ void trig_builtins(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float sn = __sinf(v);
        float cs = __cosf(v);
        out[tid] = sn * sn + cs * cs;  // should be ~1.0
    }
}

__global__ void exp_log_builtins(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = fabsf(in[tid]) + 1e-6f;
        float e = __expf(v);
        float l = __logf(e);
        out[tid] = l;  // should be ~v
    }
}

// Integer math builtins
__global__ void int_builtins(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = (unsigned int)in[tid];
        int pc = __popc(v);
        int clz = __clz(v);
        out[tid] = pc + clz;
    }
}
