// Probe: device function inlining correctness — single return path (clean),
// conditional computation without early return, and complex call-site patterns.
// Note: multi-return-inside-if is a known limitation; these probes explore
// the boundary of what inlining handles correctly.

// ------------------------------------------------------------------
// Device fn: single return, computation in branches (not early return).

__device__ int abs_diff(int a, int b) {
    int d = a - b;
    if (d < 0) d = -d;
    return d;
}

__global__ void call_abs_diff(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = abs_diff(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// Device fn called inside a conditional — call site inside if-block.

__device__ float dot2(float ax, float ay, float bx, float by) {
    return ax * bx + ay * by;
}

__global__ void cond_call(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = in[tid * 2];
        float y = in[tid * 2 + 1];
        float result;
        if (x > 0.0f) {
            result = dot2(x, y, x, y);    // |v|^2
        } else {
            result = dot2(-x, -y, x, y);  // -|v|^2
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Device fn called in loop body.

__device__ int popcount_nibble(int v) {
    // count bits in low 4 bits
    int c = 0;
    if (v & 1) c++;
    if (v & 2) c++;
    if (v & 4) c++;
    if (v & 8) c++;
    return c;
}

__global__ void loop_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int total = 0;
        for (int shift = 0; shift < 32; shift += 4) {
            total += popcount_nibble((v >> shift) & 0xF);
        }
        out[tid] = total;
    }
}

// ------------------------------------------------------------------
// Two device fns, one calls the other.

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float leaky_relu(float x, float slope) {
    return x > 0.0f ? x : slope * x;
}

__device__ float prelu(float x, float slope) {
    float r = relu(x);
    float l = leaky_relu(x, slope);
    return r + l;  // double activation for test
}

__global__ void nested_device_calls(float *out, float *in, float *slopes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = prelu(in[tid], slopes[tid]);
    }
}

// ------------------------------------------------------------------
// Device fn with loop that returns a value used in further computation.

__device__ int first_bit_set(int v) {
    for (int i = 0; i < 32; i++) {
        if ((v >> i) & 1) return i;
    }
    return -1;
}

__global__ void use_first_bit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int fb = first_bit_set(in[tid]);
        out[tid] = fb >= 0 ? (1 << fb) : 0;
    }
}

// ------------------------------------------------------------------
// Multiple calls to same device fn with different args.

__device__ int clamp_range(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__global__ void multi_clamp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = clamp_range(v,      0,  255);
        int b = clamp_range(v - 10, 0,  100);
        int c = clamp_range(v + 10, 50, 200);
        out[tid] = a + b + c;
    }
}
