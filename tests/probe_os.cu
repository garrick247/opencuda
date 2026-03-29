// Probe: high register pressure patterns, many live variables simultaneously,
// interleaved float/int registers, and complex phi-merge scenarios.

// ------------------------------------------------------------------
// High live-variable count: 8 independent accumulators across a loop.
// Tests that the register allocator assigns distinct registers to each.

__global__ void eight_accum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            s0 += v;
            s1 += v * v;
            s2 += v * v * v;
            s3 += v + 1.0f;
            s4 += v - 1.0f;
            s5 += v * 2.0f;
            s6 += v * 0.5f;
            s7 += 1.0f;
        }
        out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
        out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
    }
}

// ------------------------------------------------------------------
// Interleaved float/int in same loop: test that f and r registers don't alias.

__global__ void mixed_reg_loop(float *fout, int *iout, float *fdata, int *idata, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float fsum = 0.0f;
        int   isum = 0;
        for (int i = 0; i < n; i++) {
            fsum += fdata[i];
            isum += idata[i];
        }
        fout[0] = fsum;
        iout[0] = isum;
    }
}

// ------------------------------------------------------------------
// Long live ranges: variable defined early, used much later.
// Tests that the live range extends correctly through many instructions.

__global__ void long_live_range(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int early = data[0];      // defined at start
        int sum = 0;
        for (int i = 1; i < n; i++) {
            sum += data[i];
        }
        // early is still live here — must not have been reallocated
        out[0] = sum + early;     // uses both
    }
}

// ------------------------------------------------------------------
// Back-to-back inlines: 4 consecutive device function calls in same block.
// Each inline adds new Values; tests that register reuse doesn't cause aliasing.

__device__ float double_val(float x) { return x * 2.0f; }

__global__ void four_inlines(float *out, float a, float b, float c, float d) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float ra = double_val(a);
        float rb = double_val(b);
        float rc = double_val(c);
        float rd = double_val(d);
        out[0] = ra + rb + rc + rd;
    }
}
