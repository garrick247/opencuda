// Probe: multiple call sites for same device fn, pointer-to-pointer,
// high register pressure, and complex phi patterns.

// ------------------------------------------------------------------
// Device fn called from two different sites in same kernel.

__device__ float clamp_and_scale(float v, float lo, float hi, float scale) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v * scale;
}

__global__ void multi_call_site(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Two independent calls to same device fn
        float x = clamp_and_scale(a[tid], -1.0f, 1.0f, 2.0f);
        float y = clamp_and_scale(b[tid],  0.0f, 10.0f, 0.5f);
        out[tid] = x + y;
    }
}

// ------------------------------------------------------------------
// Device fn called inside a loop (multiple dynamic call sites).

__device__ int classify(int v) {
    if (v < 0)   return -1;
    if (v < 10)  return 0;
    if (v < 100) return 1;
    return 2;
}

__global__ void call_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            acc += classify(in[tid * 4 + i]);
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// High register pressure: 16 simultaneous live values.

__global__ void reg_pressure_16(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r0  = v + 0.0f;
        float r1  = v + 1.0f;
        float r2  = v + 2.0f;
        float r3  = v + 3.0f;
        float r4  = v + 4.0f;
        float r5  = v + 5.0f;
        float r6  = v + 6.0f;
        float r7  = v + 7.0f;
        float r8  = v * r0;
        float r9  = v * r1;
        float r10 = v * r2;
        float r11 = v * r3;
        float r12 = r8  + r4;
        float r13 = r9  + r5;
        float r14 = r10 + r6;
        float r15 = r11 + r7;
        out[tid] = r12 + r13 + r14 + r15;
    }
}

// ------------------------------------------------------------------
// Phi merge: value used after if-else with different types in arms.
// (Both arms write float, so phi is straightforward.)

__global__ void phi_after_if(float *out, float *in, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r;
        int f = flags[tid];
        if (f == 0)      r = v;
        else if (f == 1) r = v * v;
        else if (f == 2) r = sqrtf(v > 0.0f ? v : -v);
        else if (f == 3) r = 1.0f / (v + 1.0f);
        else             r = 0.0f;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Loop with phi for loop-carried float accumulator + int counter.

__global__ void dual_phi_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float sum = 0.0f;
        int   cnt = 0;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v > 0.0f) {
                sum += v;
                cnt++;
            }
        }
        out[tid] = (cnt > 0) ? (sum / (float)cnt) : 0.0f;
    }
}

// ------------------------------------------------------------------
// Phi with pointer: select different arrays based on condition.

__global__ void ptr_phi(float *out, float *a, float *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float val;
        if (sel[tid] == 0) {
            val = a[tid];
        } else {
            val = b[tid];
        }
        out[tid] = val * 2.0f;
    }
}
