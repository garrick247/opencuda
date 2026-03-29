// Probe: loop trip-count boundary (>16 must NOT unroll), char/byte ops,
// complex predicate combining, and unsigned comparisons.

// ------------------------------------------------------------------
// Trip count exactly 16 — should unroll (boundary case).

__global__ void trip16_unroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 16; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Trip count 17 — must NOT unroll (stays as runtime loop).

__global__ void trip17_noroll(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 17; i++) {
            acc += v + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Trip count 32 — definitely stays as loop.

__global__ void trip32_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 32; i++) {
            acc += v ^ i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Char/byte operations: byte extraction, packing, and arithmetic.

__global__ void byte_ops(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Extract bytes
        unsigned char b0 = (unsigned char)(v & 0xFF);
        unsigned char b1 = (unsigned char)((v >> 8) & 0xFF);
        unsigned char b2 = (unsigned char)((v >> 16) & 0xFF);
        unsigned char b3 = (unsigned char)((v >> 24) & 0xFF);
        // Swap bytes: b0↔b3, b1↔b2
        unsigned int swapped = ((unsigned int)b0 << 24)
                             | ((unsigned int)b1 << 16)
                             | ((unsigned int)b2 << 8)
                             |  (unsigned int)b3;
        out[tid] = swapped;
    }
}

// ------------------------------------------------------------------
// Signed char arithmetic — sign extension matters.

__global__ void signed_char_arith(int *out, signed char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char v = in[tid];
        // Sign extends to int
        int i = (int)v;
        int r = i * i - i + 1;   // v^2 - v + 1
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Unsigned comparison chain: multiple unsigned predicates combined.

__global__ void unsigned_cmp_chain(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // All conditions use unsigned comparisons
        int r = 0;
        if (v < 100u)          r += 1;
        if (v > 200u)          r += 2;
        if (v <= 150u)         r += 4;
        if (v >= 50u)          r += 8;
        if (v != 0u)           r += 16;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Short (int16) arithmetic via int with explicit masking.

__global__ void short_arith(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // Simulate 16-bit truncation explicitly
        short lo = (short)(v & 0xFFFF);
        short hi = (short)((v >> 16) & 0xFFFF);
        int product = (int)lo * (int)hi;
        out[tid] = product;
    }
}

// ------------------------------------------------------------------
// Predicate with negation: !pred used in branch and ternary.

__global__ void pred_negation(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = !(v > 0);           // 0 if v>0, 1 otherwise
        int b = !(v == 0);          // 0 if v==0, 1 otherwise
        int c = !(v < -10);         // 0 if v<-10, 1 otherwise
        int r = (a ? 100 : -100) + (b ? 10 : -10) + (c ? 1 : -1);
        out[tid] = r;
    }
}
