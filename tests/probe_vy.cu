// Probe: phi-node correctness under multiple loop exits, predicate reuse
// past p127, store-load in loop body (CSE safety), long dep chains,
// and aliased pointer patterns.

// ------------------------------------------------------------------
// Loop with two back-edges (natural loop + continue target).
// The phi at loop header must merge from: pre-loop init AND loop-body update.

__global__ void phi_two_back_edges(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = 0;
        for (int i = 0; i < 8; i++) {
            if (i % 2 == 0) {
                a += v;       // even: add to a
                continue;     // back edge from here (continue)
            }
            b += i;           // odd: add i to b
        }
        // a = 4*v (i=0,2,4,6), b = 1+3+5+7 = 16
        out[tid] = a + b;
    }
}

// ------------------------------------------------------------------
// Store-load in loop: must not CSE across the store.

__global__ void store_load_loop(int *out, int *buf, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        buf[tid] = tid;        // initial store
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            int v1 = buf[tid];         // load (may be stale without CSE guard)
            buf[tid] = v1 + 1;         // store (increments)
            int v2 = buf[tid];         // must reload, not CSE with v1
            sum += v2;
        }
        out[tid] = sum;
        // Each iteration: v1=tid+i, store tid+i+1, v2=tid+i+1
        // sum = (tid+1) + (tid+2) + (tid+3) + (tid+4) = 4*tid + 10
    }
}

// ------------------------------------------------------------------
// 16-deep add chain (tests value numbering / dependency tracking).

__global__ void dep_chain_16(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a0  = v + 1;
        int a1  = a0 + 2;
        int a2  = a1 + 3;
        int a3  = a2 + 4;
        int a4  = a3 + 5;
        int a5  = a4 + 6;
        int a6  = a5 + 7;
        int a7  = a6 + 8;
        int a8  = a7 + 9;
        int a9  = a8 + 10;
        int a10 = a9 + 11;
        int a11 = a10 + 12;
        int a12 = a11 + 13;
        int a13 = a12 + 14;
        int a14 = a13 + 15;
        int a15 = a14 + 16;
        out[tid] = a15;  // v + (1+2+...+16) = v + 136
    }
}

// ------------------------------------------------------------------
// Many predicates: 10 independent conditions — tests pred register pressure.

__global__ void many_predicates(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        if (v > 0)   r |= 1;
        if (v > 10)  r |= 2;
        if (v > 20)  r |= 4;
        if (v > 30)  r |= 8;
        if (v > 40)  r |= 16;
        if (v > 50)  r |= 32;
        if (v > 60)  r |= 64;
        if (v > 70)  r |= 128;
        if (v > 80)  r |= 256;
        if (v > 90)  r |= 512;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Aliased writes: two pointers to the same array, interleaved access.

__global__ void aliased_write(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid + 1 < n) {
        int *p = in + tid;
        int *q = in + tid + 1;  // overlaps with p's neighbor
        int v1 = *p;
        *q = v1 * 2;            // write through q
        int v2 = *p;            // re-read p — different address, safe
        out[tid] = v1 + v2;     // 2 * in[tid]
    }
}

// ------------------------------------------------------------------
// Accumulator pattern: loop accumulates into same variable, no CSE.

__global__ void loop_accum(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int i = 0; i < 8; i++) {
            float v = in[tid * 8 + i];
            acc = acc + v;   // no FMA — separate add
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Two loop-carried variables that depend on each other (swap pattern).

__global__ void loop_swap(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = v, b = v + 1;
        for (int i = 0; i < 4; i++) {
            int tmp = a;
            a = b;
            b = tmp + b;    // fib-like: b = old_a + old_b
        }
        out[tid] = a + b;
        // i=0: a=v+1, b=v+(v+1)=2v+1
        // i=1: a=2v+1, b=(v+1)+(2v+1)=3v+2
        // i=2: a=3v+2, b=(2v+1)+(3v+2)=5v+3
        // i=3: a=5v+3, b=(3v+2)+(5v+3)=8v+5
        // final: a+b = 13v+8
    }
}

// ------------------------------------------------------------------
// Conditional inside loop: loop-carried var set in both branches.

__global__ void cond_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int x = 0;
        for (int i = 0; i < 8; i++) {
            if (i < v % 4) {
                x += i * 2;
            } else {
                x += i;
            }
        }
        out[tid] = x;
    }
}
