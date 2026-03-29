// Probe: conditional struct field mutation inside inline (no early return),
//        two struct vars of same type in same kernel

struct Stats { float sum; int count; };

// Conditionally updates fields — if-only, no early return
// The false path must NOT use the if-true path's updated registers
__device__ Stats filter_update(Stats s, float x, float threshold) {
    if (x > threshold) {
        s.sum += x;
        s.count++;
    }
    return s;
}

// Two conditionally-updated fields, different conditions
__device__ Stats dual_filter(Stats s, float x, float lo, float hi) {
    if (x >= lo) { s.sum += x; }
    if (x <= hi) { s.count++; }
    return s;
}

// Two struct vars of same type (Stats) in same kernel — field key isolation
__device__ Stats step_one(Stats s, float x) {
    s.sum += x;
    s.count++;
    return s;
}

// Expected: s_a accumulates x[i], s_b accumulates x[n-1-i]
// Both independently updated via same inline function
__global__ void two_accumulators(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s_a; s_a.sum = 0.0f; s_a.count = 0;
        Stats s_b; s_b.sum = 0.0f; s_b.count = 0;
        for (int i = 0; i < n; i++) {
            s_a = step_one(s_a, in[i]);
            s_b = step_one(s_b, in[n - 1 - i]);
        }
        out[0] = s_a.sum;
        out[1] = (float)s_a.count;
        out[2] = s_b.sum;
        out[3] = (float)s_b.count;
    }
}

// Threshold filter: count and sum only values > threshold
// Expected: sum = sum(in[i] where in[i] > 0.5), count = n_above
__global__ void threshold_filter(float *out, float *in, float threshold, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s; s.sum = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            s = filter_update(s, in[i], threshold);
        }
        out[0] = s.sum;
        out[1] = (float)s.count;
    }
}

// Dual filter: sum accumulated if x >= lo, count if x <= hi
__global__ void dual_filter_kernel(float *out, float *in,
                                    float lo, float hi, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s; s.sum = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            s = dual_filter(s, in[i], lo, hi);
        }
        out[0] = s.sum;
        out[1] = (float)s.count;
    }
}
