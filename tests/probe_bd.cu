// Probe: texture memory patterns and special CUDA types
// - cudaTextureObject_t (opaque handle, parse as long long or uint64)
// - tex1Dfetch style calls
// - half2 arithmetic  
// - int4, float4 struct-like types
// - __half type usage

// Simulate texture fetch with regular pointer (tests parsing, not semantics)
__global__ void tex_sim(float *out, float *texdata, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // tex1D-style: just use regular array since we can't have textures
        float v = texdata[tid % n];
        out[tid] = v * 2.0f;
    }
}

// int4-like struct access
struct int4_t {
    int x, y, z, w;
};

__global__ void int4_ops(int *out, int4_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int4_t v = in[tid];
        out[tid] = v.x + v.y + v.z + v.w;
    }
}

// Compound operation with multiple types
__global__ void mixed_arith(float *fout, int *iout, float *fin, int *iin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = fin[tid];
        int   i = iin[tid];
        float fi = f + (float)i;
        int   if2 = (int)f + i;
        fout[tid] = fi;
        iout[tid] = if2;
    }
}
