// Probe: parallel-copy sequencing stress test
// All three kernels exercise inter-field dependencies in struct-updating inlines
// that require the _loop_writeback sequencing fix (v0.27) to be correct.

// ------------------------------------------------------------------
// Three-field rotation: s.c = s.b; s.b = s.a; s.a = v
// Each field takes the value of the previous field, and s.a gets v.
// After N iterations: s.a = data[N-1], s.b = data[N-2], s.c = data[N-3].
// Tests that 3-step parallel copy chain is sequenced correctly.

struct Rot3 { float a; float b; float c; };

__device__ Rot3 rot3_push(Rot3 s, float v) {
    s.c = s.b;
    s.b = s.a;
    s.a = v;
    return s;
}

__global__ void triple_rotation(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Rot3 s; s.a = 0.0f; s.b = 0.0f; s.c = 0.0f;
        for (int i = 0; i < n; i++) {
            s = rot3_push(s, data[i]);
        }
        out[0] = s.a;
        out[1] = s.b;
        out[2] = s.c;
    }
}

// ------------------------------------------------------------------
// Four-field shift accumulate:
// Each field accumulates the VALUE of the field above it.
// s.d += s.c; s.c += s.b; s.b += s.a; s.a += v
// Tests 4-step dependency chain where each writeback reads the
// entry value of an adjacent field.

struct Acc4 { float a; float b; float c; float d; };

__device__ Acc4 acc4_step(Acc4 s, float v) {
    s.d += s.c;
    s.c += s.b;
    s.b += s.a;
    s.a += v;
    return s;
}

__global__ void shift_accum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc4 s; s.a = 0.0f; s.b = 0.0f; s.c = 0.0f; s.d = 0.0f;
        for (int i = 0; i < n; i++) {
            s = acc4_step(s, data[i]);
        }
        out[0] = s.a;
        out[1] = s.b;
        out[2] = s.c;
        out[3] = s.d;
    }
}

// ------------------------------------------------------------------
// Mixed read-write and rotation:
// One field is read-modified-written (accumulate), two fields rotate.
// s.sum += v; s.prev2 = s.prev1; s.prev1 = v
// Tests that the rotation dependency (prev2 := prev1) is correctly
// sequenced while the accumulation (sum += v) is independent.

struct RunRot { float sum; float prev1; float prev2; };

__device__ RunRot runrot_update(RunRot s, float v) {
    s.sum   += v;
    s.prev2  = s.prev1;
    s.prev1  = v;
    return s;
}

__global__ void run_with_rotation(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        RunRot s; s.sum = 0.0f; s.prev1 = 0.0f; s.prev2 = 0.0f;
        for (int i = 0; i < n; i++) {
            s = runrot_update(s, data[i]);
        }
        out[0] = s.sum;
        out[1] = s.prev1;
        out[2] = s.prev2;
    }
}
