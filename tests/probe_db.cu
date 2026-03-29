// Probe: CUDA-specific intrinsics and builtins
// - __float_as_int / __int_as_float (type punning)  
// - __fdividef (fast float divide)
// - __expf, __logf, __sinf, __cosf (fast math variants)
// - min/max as C functions (not CUDA intrinsics)
// - fminf/fmaxf 

__global__ void fast_math(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = sqrtf(fabsf(v));
        float s = fminf(r, 100.0f);
        float t = fmaxf(s, 0.0f);
        out[tid] = t;
    }
}

// Type punning intrinsics
__global__ void type_pun(int *iout, float *fin, float *fout, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float fv = fin[tid];
        int iv = iin[tid];
        // Just cast-based punning (we don't have __float_as_int)
        float fc = (float)iv;
        int ic = (int)fv;
        fout[tid] = fc;
        iout[tid] = ic;
    }
}

// min/max over arrays
__global__ void array_minmax(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < 1) {
        int mn = in[0], mx = in[0];
        for (int i = 1; i < n; i++) {
            int v = in[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        out[0] = mn;
        out[1] = mx;
    }
}
