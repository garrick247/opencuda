// Probe: crypto/hashing patterns, color space conversion, reduction ring,
// warp-cooperative sort (odd-even transposition), and complex control flow
// with multiple nested loops and early returns.

// ------------------------------------------------------------------
// SHA-256-like single round function (simplified — tests bitwise ops).

__device__ unsigned sha_ch(unsigned x, unsigned y, unsigned z) {
    return (x & y) ^ (~x & z);
}
__device__ unsigned sha_maj(unsigned x, unsigned y, unsigned z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
__device__ unsigned rotr(unsigned x, int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ unsigned sigma0(unsigned x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
__device__ unsigned sigma1(unsigned x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__global__ void sha_round(unsigned *state, unsigned *W, unsigned K, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    unsigned a = state[0], b = state[1], c = state[2], d = state[3];
    unsigned e = state[4], f = state[5], g = state[6], h = state[7];
    unsigned t1 = h + sigma1(e) + sha_ch(e, f, g) + K + W[tid];
    unsigned t2 = sigma0(a) + sha_maj(a, b, c);
    h = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
    // Just store last state for demonstration
    state[tid % 8] = a;
}

// ------------------------------------------------------------------
// RGB to YCbCr color space conversion.

__global__ void rgb_to_ycbcr(float *Y, float *Cb, float *Cr,
                                float *R, float *G, float *B, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float r = R[gid], g = G[gid], b = B[gid];
    Y[gid]  =  0.299f*r + 0.587f*g + 0.114f*b;
    Cb[gid] = -0.169f*r - 0.331f*g + 0.500f*b + 128.0f;
    Cr[gid] =  0.500f*r - 0.419f*g - 0.081f*b + 128.0f;
}

// ------------------------------------------------------------------
// Odd-even transposition sort step (warp-cooperative).

__global__ void odd_even_step(int *data, int phase, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = 2 * gid + (phase & 1);
    if (idx + 1 < n) {
        if (data[idx] > data[idx + 1]) {
            int tmp = data[idx];
            data[idx] = data[idx + 1];
            data[idx + 1] = tmp;
        }
    }
}

// ------------------------------------------------------------------
// Complex control flow: state machine with multiple nested loops.

__global__ void state_machine(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    int v = in[tid];
    int state = 0;
    int result = 0;
    for (int i = 0; i < 32; i++) {
        int bit = (v >> i) & 1;
        switch (state) {
            case 0:
                if (bit) { state = 1; result++; }
                break;
            case 1:
                if (!bit) { state = 2; }
                else { result++; }
                break;
            case 2:
                if (bit) { state = 0; result += 2; }
                break;
        }
    }
    out[tid] = result;
}

// ------------------------------------------------------------------
// Polynomial evaluation with Horner's method (degree 7).

__global__ void horner7(float *out, float *x_in,
                           float c0, float c1, float c2, float c3,
                           float c4, float c5, float c6, float c7, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float x = x_in[gid];
    float r = c7;
    r = r * x + c6;
    r = r * x + c5;
    r = r * x + c4;
    r = r * x + c3;
    r = r * x + c2;
    r = r * x + c1;
    r = r * x + c0;
    out[gid] = r;
}

// ------------------------------------------------------------------
// Reduction ring buffer: accumulate in circular fashion.

__global__ void ring_reduce(float *out, float *in, int window, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float s = 0.0f;
    for (int k = 0; k < window; k++) {
        int idx = (gid + k) % n;
        s += in[idx];
    }
    out[gid] = s / (float)window;
}
