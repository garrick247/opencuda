// Probe: byte manipulation, endian swap, bit extraction, manual
// memory copy patterns, and packed integer formats.

// ------------------------------------------------------------------
// Manual byte copy (memcpy pattern).

__global__ void byte_copy(unsigned char *dst, unsigned char *src, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

// ------------------------------------------------------------------
// Endian swap: 32-bit big-endian to little-endian.

__global__ void endian_swap32(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = ((v & 0xFF000000u) >> 24)
                 | ((v & 0x00FF0000u) >>  8)
                 | ((v & 0x0000FF00u) <<  8)
                 | ((v & 0x000000FFu) << 24);
    }
}

// ------------------------------------------------------------------
// Endian swap: 64-bit.

__global__ void endian_swap64(unsigned long long *out, unsigned long long *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long v = in[tid];
        out[tid] = ((v & 0xFF00000000000000ULL) >> 56)
                 | ((v & 0x00FF000000000000ULL) >> 40)
                 | ((v & 0x0000FF0000000000ULL) >> 24)
                 | ((v & 0x000000FF00000000ULL) >>  8)
                 | ((v & 0x00000000FF000000ULL) <<  8)
                 | ((v & 0x0000000000FF0000ULL) << 24)
                 | ((v & 0x000000000000FF00ULL) << 40)
                 | ((v & 0x00000000000000FFULL) << 56);
    }
}

// ------------------------------------------------------------------
// Nibble extraction.

__global__ void nibble_extract(unsigned char *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid * 8 + 0] = (v >>  0) & 0xF;
        out[tid * 8 + 1] = (v >>  4) & 0xF;
        out[tid * 8 + 2] = (v >>  8) & 0xF;
        out[tid * 8 + 3] = (v >> 12) & 0xF;
        out[tid * 8 + 4] = (v >> 16) & 0xF;
        out[tid * 8 + 5] = (v >> 20) & 0xF;
        out[tid * 8 + 6] = (v >> 24) & 0xF;
        out[tid * 8 + 7] = (v >> 28) & 0xF;
    }
}

// ------------------------------------------------------------------
// Bit field pack/unpack (without C bitfield syntax — manual shifts).

// Pack: r(8) g(8) b(8) a(8) into one uint.
__global__ void rgba_pack(unsigned int *out,
                           unsigned char *r, unsigned char *g,
                           unsigned char *b, unsigned char *a, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ((unsigned int)a[tid] << 24)
                 | ((unsigned int)b[tid] << 16)
                 | ((unsigned int)g[tid] <<  8)
                 |  (unsigned int)r[tid];
    }
}

// Unpack rgba.
__global__ void rgba_unpack(unsigned char *r, unsigned char *g,
                              unsigned char *b, unsigned char *a,
                              unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        r[tid] = (unsigned char)( v        & 0xFF);
        g[tid] = (unsigned char)((v >>  8) & 0xFF);
        b[tid] = (unsigned char)((v >> 16) & 0xFF);
        a[tid] = (unsigned char)((v >> 24) & 0xFF);
    }
}

// ------------------------------------------------------------------
// Gray code encode.

__global__ void gray_encode(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        out[tid] = v ^ (v >> 1);
    }
}

// ------------------------------------------------------------------
// Saturating add (no intrinsic — manual clamp).

__global__ void sat_add_u8(unsigned char *out, unsigned char *a, unsigned char *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = (int)a[tid] + (int)b[tid];
        out[tid] = (unsigned char)(sum > 255 ? 255 : sum);
    }
}
