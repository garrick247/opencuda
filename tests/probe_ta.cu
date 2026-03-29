// Probe: short and char arithmetic, unsigned loop counters, chained
// ternary inside loop, and phi with 3+ predecessors.

// ------------------------------------------------------------------
// short arithmetic.

__global__ void short_arith(short *out, short *a, short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        short x = a[tid];
        short y = b[tid];
        short r = (short)((x * y - x + y) & 0x7FFF);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// char arithmetic (signed).

__global__ void char_arith(signed char *out, signed char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char v = in[tid];
        signed char r = (signed char)((v * v - 1) & 0x7F);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// unsigned loop counter (u32 for-loop).

__global__ void uint_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = (unsigned int)in[tid];
        unsigned int acc = 0;
        for (unsigned int i = 0u; i < 16u; i++) {
            acc += v >> i;
        }
        out[tid] = (int)acc;
    }
}

// ------------------------------------------------------------------
// Chained ternary inside loop body.

__global__ void ternary_chain_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            int x = v + i;
            int clamped = x < 0 ? 0 : x > 100 ? 100 : x > 50 ? x - 25 : x;
            acc += clamped;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Phi with 4 predecessors (if/else-if/else-if/else).

__global__ void phi_4way(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (v < -100)      r = -3;
        else if (v < 0)    r = -1;
        else if (v < 100)  r = 1;
        else               r = 3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Phi with 5 predecessors (switch-fall-through emulation).

__global__ void phi_5way(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 5;
        int r;
        if (v == 0)      r = 100;
        else if (v == 1) r = 200;
        else if (v == 2) r = 300;
        else if (v == 3) r = 400;
        else             r = 500;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// unsigned short used as array index.

__global__ void ushort_index(float *out, float *in, unsigned short *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned short i = idx[tid];
        out[tid] = in[(int)i] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Mixed signed/unsigned comparison (classic C pitfall).

__global__ void sign_mismatch_cmp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        unsigned int uv = (unsigned int)v;
        // signed/unsigned comparison — uv >= 0u is always true
        int r = (uv >= 0u) ? v : -v;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Short loaded, widened, computed, narrowed back.

__global__ void short_widen_narrow(short *out, short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = (int)in[tid];  // widen
        v = v * v - v + 7;    // compute in int
        out[tid] = (short)(v & 0xFFFF);  // narrow back
    }
}
