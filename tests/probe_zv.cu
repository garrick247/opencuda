// Probe: audio/signal processing — FIR filter, IIR biquad, pitch detection
// via autocorrelation, envelope follower, fast Walsh-Hadamard transform,
// and overlap-add convolution block.

// ------------------------------------------------------------------
// FIR filter with shared-memory coefficient caching.

__constant__ float fir_taps[64];

__global__ void fir_filter(float *out, float *in, int tap_len, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float s = 0.0f;
    for (int k = 0; k < tap_len && k < 64; k++) {
        int idx = gid - k;
        if (idx >= 0) s += in[idx] * fir_taps[k];
    }
    out[gid] = s;
}

// ------------------------------------------------------------------
// IIR biquad filter (serial per-sample, parallel per-channel).

__global__ void biquad_filter(float *out, float *in,
                                 float b0, float b1, float b2,
                                 float a1, float a2, int n) {
    int ch = blockIdx.x;  // one block per channel
    float x1 = 0.0f, x2 = 0.0f;
    float y1 = 0.0f, y2 = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = in[ch * n + i];
        float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        out[ch * n + i] = y;
        x2 = x1; x1 = x;
        y2 = y1; y1 = y;
    }
}

// ------------------------------------------------------------------
// Autocorrelation for pitch detection.

__global__ void autocorrelation(float *out, float *in, int window, int max_lag, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= max_lag) return;
    int lag = gid;
    float s = 0.0f;
    for (int i = 0; i < window - lag; i++) {
        s += in[i] * in[i + lag];
    }
    out[lag] = s;
}

// ------------------------------------------------------------------
// Envelope follower (attack/release).

__global__ void envelope(float *out, float *in, float attack, float release, int n) {
    int ch = blockIdx.x;
    float env = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(in[ch * n + i]);
        float coeff = (v > env) ? attack : release;
        env = env + coeff * (v - env);
        out[ch * n + i] = env;
    }
}

// ------------------------------------------------------------------
// Fast Walsh-Hadamard transform (in-place, power-of-2 size).

__global__ void fwht_step(float *data, int half_size, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n / 2) return;
    int group = gid / half_size;
    int pair  = gid % half_size;
    int i = group * 2 * half_size + pair;
    int j = i + half_size;
    float a = data[i];
    float b = data[j];
    data[i] = a + b;
    data[j] = a - b;
}

// ------------------------------------------------------------------
// Overlap-add: accumulate windowed blocks.

__global__ void overlap_add(float *out, float *blocks, int block_size,
                               int hop_size, int n_blocks) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        int offset = b * hop_size;
        int local = gid - offset;
        if (local >= 0 && local < block_size) {
            s += blocks[b * block_size + local];
        }
    }
    out[gid] = s;
}
