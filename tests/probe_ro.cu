// Probe: do-while with break, chained ternary, pointer arithmetic with
// non-4-byte strides (short*, char*), device fn returning struct,
// and global array of shorts.

// ------------------------------------------------------------------
// do-while with break: early exit from do-while.

__global__ void do_while_break(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int count = 0;
        do {
            if (v <= 0) break;
            v /= 2;
            count++;
        } while (v > 1);
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Chained ternary: a ? b : c ? d : e.

__global__ void chained_ternary(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r = v > 100 ? 3 :
                v > 50  ? 2 :
                v > 0   ? 1 : 0;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Pointer arithmetic with short stride: short* indexing.

__global__ void short_stride(short *out, short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        short v = in[tid];
        out[tid] = v * 2;
    }
}

// ------------------------------------------------------------------
// Pointer arithmetic with char stride.

__global__ void char_stride(signed char *out, signed char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed char v = in[tid];
        out[tid] = (signed char)(v + 1);
    }
}

// ------------------------------------------------------------------
// Global array of shorts.

__device__ short g_short_lut[16];

__global__ void short_lut_kernel(int *out, int *idx, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = idx[tid] & 15;
        out[tid] = (int)g_short_lut[i];
    }
}

// ------------------------------------------------------------------
// Device function returning a struct (small struct).

struct Pair {
    float min;
    float max;
};

__device__ Pair minmax(float a, float b) {
    Pair p;
    p.min = a < b ? a : b;
    p.max = a > b ? a : b;
    return p;
}

__global__ void minmax_kernel(float *omin, float *omax, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair p = minmax(a[tid], b[tid]);
        omin[tid] = p.min;
        omax[tid] = p.max;
    }
}

// ------------------------------------------------------------------
// Short accumulate: scan short array and sum as int.

__global__ void short_sum(int *out, short *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += (int)in[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Mixed short/int: compare short with int constant.

__global__ void short_compare(int *out, short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        short v = in[tid];
        // C promotion: short vs int → both become int
        if (v > 32767) {
            out[tid] = 1;
        } else if (v < -32768) {
            out[tid] = -1;
        } else {
            out[tid] = 0;
        }
    }
}
