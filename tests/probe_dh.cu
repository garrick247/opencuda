// Probe: Tricky cases involving device function with OUT parameters
// - Device func modifying float* and int* output params simultaneously
// - Device func with mixed pass-by-value and pass-by-pointer
// - Using return value AND output params from same call (not possible but test pattern)

__device__ void decompose(float x, int *whole_part, float *frac_part) {
    *whole_part = (int)x;
    *frac_part = x - (float)(*whole_part);
}

__device__ void minmax_pair(float a, float b, float *mn, float *mx) {
    if (a < b) { *mn = a; *mx = b; }
    else { *mn = b; *mx = a; }
}

__global__ void mixed_params(float *fout, int *iout, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int w;
        float frac;
        decompose(v, &w, &frac);
        
        float mn, mx;
        minmax_pair(v, frac, &mn, &mx);
        
        iout[tid] = w;
        fout[tid] = mn + mx;
    }
}
