// Probe: two sequential loops on same struct, inline fn with loop body,
//        struct copy on declaration

struct Stats { float sum, sum2; int count; };

__device__ Stats step_s(Stats s, float x) {
    s.sum  += x;
    s.sum2 += x * x;
    s.count++;
    return s;
}

// Inline function that itself contains a for-loop
__device__ Stats batch_step(Stats s, float *in, int start, int len) {
    for (int k = 0; k < len; k++) {
        s.sum  += in[start + k];
        s.sum2 += in[start + k] * in[start + k];
        s.count++;
    }
    return s;
}

// Two sequential loops: loop1 builds stats, loop2 reads stats to output
__global__ void two_seq_loops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s; s.sum = 0.0f; s.sum2 = 0.0f; s.count = 0;
        // Loop 1: accumulate
        for (int i = 0; i < n; i++) {
            s = step_s(s, in[i]);
        }
        // Between loops: use struct fields (mean computation)
        float mean = (s.count > 0) ? s.sum / (float)s.count : 0.0f;
        // Loop 2: count deviations above mean
        int above = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > mean) above++;
        }
        out[0] = s.sum;
        out[1] = s.sum2;
        out[2] = (float)s.count;
        out[3] = mean;
        out[4] = (float)above;
    }
}

// Struct copy on declaration: Stats b = a; then modify b independently
__global__ void struct_copy_decl(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats a; a.sum = 0.0f; a.sum2 = 0.0f; a.count = 0;
        for (int i = 0; i < n; i++) {
            a = step_s(a, in[i]);
        }
        Stats b = a;           // copy on declaration
        b.sum  += 100.0f;      // modify b independently
        b.count = 999;
        out[0] = a.sum;        // a unchanged
        out[1] = (float)a.count;
        out[2] = b.sum;        // b = a.sum + 100
        out[3] = (float)b.count; // b.count = 999
    }
}

// Inline with loop body called from outer loop (nested loops via inline)
// Note: batch_step(s, in, i*2, 2) processes 2 elements per outer iteration
__global__ void inline_with_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats s; s.sum = 0.0f; s.sum2 = 0.0f; s.count = 0;
        int pairs = n / 2;
        for (int i = 0; i < pairs; i++) {
            s = batch_step(s, in, i * 2, 2);
        }
        out[0] = s.sum;
        out[1] = s.sum2;
        out[2] = (float)s.count;
    }
}
