// Nasty: int loop counter, float accumulator, half load, all live simultaneously.
// Tests per-type register pressure and type-promotion consistency under load.
__global__ void mixed_accum(half* hdata, float* fdata, float* out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float f_sum = 0.0f;
    half h_sum = 0.0f;
    int i_count = 0;

    for (int i = 0; i < n; i++) {
        half h = hdata[i];
        float  f = fdata[i];

        h_sum = h_sum + h;      // half accumulator
        f_sum = f_sum + f;      // float accumulator
        i_count = i_count + 1;  // int counter

        if (i_count % 4 == 0) {
            f_sum = f_sum + h_sum;   // promote half to float
            h_sum = 0.0f;            // reset half accumulator
        }
    }
    // All three live at the store
    out[tid] = f_sum + i_count;
}
