// Probe: register allocator stress — many simultaneous live values,
// values live across loop back-edges, and spill-heavy patterns.

// ------------------------------------------------------------------
// 32 live values simultaneously (stress test naive allocator).

__global__ void live32(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // 32 values alive simultaneously before the store
        float r00 = v + 0.0f;  float r01 = v + 1.0f;
        float r02 = v + 2.0f;  float r03 = v + 3.0f;
        float r04 = v + 4.0f;  float r05 = v + 5.0f;
        float r06 = v + 6.0f;  float r07 = v + 7.0f;
        float r08 = v + 8.0f;  float r09 = v + 9.0f;
        float r10 = v + 10.0f; float r11 = v + 11.0f;
        float r12 = v + 12.0f; float r13 = v + 13.0f;
        float r14 = v + 14.0f; float r15 = v + 15.0f;
        float r16 = v + 16.0f; float r17 = v + 17.0f;
        float r18 = v + 18.0f; float r19 = v + 19.0f;
        float r20 = v + 20.0f; float r21 = v + 21.0f;
        float r22 = v + 22.0f; float r23 = v + 23.0f;
        float r24 = v + 24.0f; float r25 = v + 25.0f;
        float r26 = v + 26.0f; float r27 = v + 27.0f;
        float r28 = v + 28.0f; float r29 = v + 29.0f;
        float r30 = v + 30.0f; float r31 = v + 31.0f;
        out[tid] = r00 + r01 + r02 + r03 + r04 + r05 + r06 + r07
                 + r08 + r09 + r10 + r11 + r12 + r13 + r14 + r15
                 + r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23
                 + r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31;
    }
}

// ------------------------------------------------------------------
// Values live across loop back-edges (allocator must extend ranges).

__global__ void live_across_backedge(float *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        // a, b, c defined before loop — must stay live through all iterations
        float a = v + 1.0f;
        float b = v * 2.0f;
        float c = v - 1.0f;
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            // a, b, c are loop-invariant and must stay live
            acc = acc + a * b - c + (float)i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Interleaved use: values defined early, used late.

__global__ void interleaved_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Define many values early
        int d0 = v & 0xFF;
        int d1 = (v >> 8) & 0xFF;
        int d2 = (v >> 16) & 0xFF;
        int d3 = (v >> 24) & 0xFF;

        // Do other computation between def and use
        int sum = v + d0;
        sum = sum * d1 + d2;
        sum = sum - d3 + d0;

        // Use all original values at the end
        out[tid] = d0 + d1 + d2 + d3 + sum;
    }
}

// ------------------------------------------------------------------
// Multiple accumulators in loop (all live simultaneously).

__global__ void multi_accum(float *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
        for (int i = 0; i < k; i++) {
            float v = in[tid * k + i];
            acc0 += v;
            acc1 += v * v;
            acc2 += v * v * v;
            acc3 += 1.0f / (v + 1.0f);
            acc4 += sqrtf(v > 0.0f ? v : -v);
            acc5 += v > 0.0f ? v : -v;
            acc6 += (float)(i & 1) * v;
            acc7 += (float)(i >> 1) * v;
        }
        out[tid * 8 + 0] = acc0;
        out[tid * 8 + 1] = acc1;
        out[tid * 8 + 2] = acc2;
        out[tid * 8 + 3] = acc3;
        out[tid * 8 + 4] = acc4;
        out[tid * 8 + 5] = acc5;
        out[tid * 8 + 6] = acc6;
        out[tid * 8 + 7] = acc7;
    }
}

// ------------------------------------------------------------------
// Long def-use chain (forces register to stay live through many ops).

__global__ void long_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v + 1;      // a defined here
        int b = a * 2;
        int c = b + 3;
        int d = c - 1;
        int e = d * d;
        int f = e + a;      // a used much later
        int g = f / 2;
        int h = g ^ d;      // d used much later
        int result = h + a + d;  // both a and d used at end
        out[tid] = result;
    }
}
