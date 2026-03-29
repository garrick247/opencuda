// Probe: register allocation stress tests — long-lived values across complex
// control flow, values that cross back-edges at multiple loop levels,
// and patterns that could expose register aliasing bugs.

// ------------------------------------------------------------------
// Value defined before loop, modified inside, used after loop exit.
// Tests back-edge liveness extension in _build_alloc_map.

__global__ void loop_live_across(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];         // defined before loop
        int acc = 0;             // defined before loop
        int count = 0;           // defined before loop

        for (int i = 0; i < 8; i++) {
            acc += v * i;        // v must stay live for all 8 iters
            if (v > acc) count++;
        }

        out[tid] = v + acc + count;  // all three used after loop
    }
}

// ------------------------------------------------------------------
// Outer value live across inner loop.

__global__ void outer_live_inner(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];    // outer value
        int result = 0;

        for (int i = 0; i < 4; i++) {
            int inner = 0;
            for (int j = 0; j < 4; j++) {
                inner += v + i + j;  // v must be live through inner loop
            }
            result += inner;
        }
        out[tid] = result + v;  // v used after both loops
    }
}

// ------------------------------------------------------------------
// Multiple values live across multiple iterations with different lifetimes.

__global__ void multi_live_lifetimes(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float a = in[tid];        // live entire function
        float b = a * 2.0f;       // live through first loop
        float c = a + 1.0f;       // live through second loop
        float sum_b = 0.0f, sum_c = 0.0f;

        // Loop 1: uses a and b
        for (int i = 0; i < 4; i++) {
            sum_b += (a + b) * (float)i;
        }

        // Loop 2: uses a and c (b dead after this point)
        for (int i = 0; i < 4; i++) {
            sum_c += (a + c) * (float)(4 - i);
        }

        out[tid] = sum_b + sum_c + a;  // a still live
    }
}

// ------------------------------------------------------------------
// Value escapes via pointer after loop modification.

__global__ void ptr_escape_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int local[8];
        for (int i = 0; i < 8; i++) {
            local[i] = in[(tid + i) % n];
        }
        // After the loop, values are in local array.
        // Sum them
        int s = 0;
        for (int i = 0; i < 8; i++) s += local[i];
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Condition-dependent register live range extension.

__global__ void cond_live_extension(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = v * 3;          // x defined early
        int y = 0;

        if (v > 10) {
            y = x * 2;          // x used in first branch
        } else {
            y = x + 5;          // x used in second branch
        }

        // x still used after the if-else
        out[tid] = x + y;       // x must still be live here
    }
}

// ------------------------------------------------------------------
// Predicate register reuse across long span.

__global__ void pred_long_live(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int pred1 = (v > 0.0f);     // predicate defined early
        int pred2 = (v < 100.0f);   // second predicate

        float r = 0.0f;
        for (int i = 0; i < 8; i++) {
            r += (float)i;
            // pred1, pred2 don't change but must stay live through loop
        }

        if (pred1 && pred2) {
            out[tid] = r + v;
        } else {
            out[tid] = r;
        }
    }
}

// ------------------------------------------------------------------
// High live-register count at a single point (register pressure stress).

__global__ void high_live_at_point(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // 12 live simultaneously before the single write
        int r0  = v + 0;
        int r1  = v + 1;
        int r2  = v + 2;
        int r3  = v + 3;
        int r4  = v + 4;
        int r5  = v + 5;
        int r6  = v + 6;
        int r7  = v + 7;
        int r8  = v + 8;
        int r9  = v + 9;
        int r10 = v + 10;
        int r11 = v + 11;
        // All 12 + v are live at the sum
        out[tid] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11 + v;
    }
}

// ------------------------------------------------------------------
// Float accumulation loop with conditional inside: phi at loop header.

__global__ void float_loop_phi(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        float scale = 1.0f;
        for (int i = 0; i < 8; i++) {
            float v = in[(tid + i) % n];
            if (v > 0.0f) {
                acc += v * scale;
                scale *= 0.9f;      // scale decremented in "hot" path
            } else {
                acc -= 0.1f;
                // scale unchanged in "cold" path
            }
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Register pressure from device fn return + continued use.

__device__ int compute_offset(int base, int stride, int idx) {
    return base + stride * idx;
}

__global__ void register_across_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];         // live across call
        int b = a + 1;           // live across call
        int c = a + 2;           // live across call
        int off = compute_offset(a, b, c);  // inline call
        // a, b, c must still be live after the call
        out[tid] = off + a + b + c;
    }
}
