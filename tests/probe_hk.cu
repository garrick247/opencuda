// Probe: #pragma unroll with explicit count, #pragma unroll (no count),
// #pragma once, other pragmas silently consumed

#pragma once

__global__ void unroll_explicit(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            sum += in[tid * 4 + i];
        }
        out[tid] = sum;
    }
}

__global__ void unroll_default(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sum += in[tid * 8 + i];
        }
        out[tid] = sum;
    }
}

// #pragma nounroll
__global__ void nounroll_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        #pragma nounroll
        for (int i = 0; i < n; i++) {
            s += in[i];
        }
        out[tid] = s;
    }
}
