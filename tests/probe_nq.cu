// Probe: switch dispatch inside loop, post-inline field as argument, adaptive stride

// ------------------------------------------------------------------
// Switch dispatch: loop body switches on (i % 3), dispatching to
// three different struct-updating inlines.  Each case updates a
// different field of Accum3; the other two fields must stay live
// through the case blocks they skip.

struct Accum3 { float a; float b; float c; };

__device__ Accum3 inc_a(Accum3 s, float v) {
    s.a += v;
    return s;
}

__device__ Accum3 inc_b(Accum3 s, float v) {
    s.b += v * 2.0f;
    return s;
}

__device__ Accum3 inc_c(Accum3 s, float v) {
    s.c += v * v;
    return s;
}

__global__ void switch_dispatch_loop(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Accum3 s; s.a = 0.0f; s.b = 0.0f; s.c = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            switch (i % 3) {
                case 0: s = inc_a(s, v); break;
                case 1: s = inc_b(s, v); break;
                case 2: s = inc_c(s, v); break;
            }
        }
        out[0] = s.a;
        out[1] = s.b;
        out[2] = s.c;
    }
}

// ------------------------------------------------------------------
// Post-inline field as argument: after scale_pair updates p, the
// field p.weight (now the POST-scale value) is passed as the delta
// to shift_pair.  Tests that the parser binds p.weight to the
// updated register, not the stale pre-scale register.

struct RunPair { float val; float weight; };

__device__ RunPair scale_pair(RunPair p, float factor) {
    p.val    *= factor;
    p.weight *= factor;
    return p;
}

__device__ RunPair shift_pair(RunPair p, float delta) {
    p.val    += delta;
    p.weight += delta * 0.5f;
    return p;
}

__global__ void field_chain_arg(float *out, float *data, int n, float init_w) {
    int tid = threadIdx.x;
    if (tid == 0) {
        RunPair p; p.val = 0.0f; p.weight = init_w;
        for (int i = 0; i < n; i++) {
            p = scale_pair(p, data[i]);
            p = shift_pair(p, p.weight);
        }
        out[0] = p.val;
        out[1] = p.weight;
    }
}

// ------------------------------------------------------------------
// Adaptive stride: loop-carried `step` is updated each iteration
// by a ternary inline.  Tests that:
// (a) LICM correctly does NOT hoist step (it changes every iteration)
// (b) the ternary output register is live-extended across the back edge
// (c) accum and step both survive correctly as loop-carried scalars

__device__ float adapt_step(float step, float val) {
    return (val > 0.0f) ? step * 2.0f : step * 0.5f;
}

__global__ void adaptive_stride(float *out, float *data, int n, float init_step) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float accum = 0.0f;
        float step  = init_step;
        for (int i = 0; i < n; i++) {
            accum += data[i] * step;
            step   = adapt_step(step, accum);
        }
        out[0] = accum;
        out[1] = step;
    }
}
