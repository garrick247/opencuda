// Probe: bool type arithmetic, char* in printf, device fn with many params,
// output-param pointer write, unsigned char/short arithmetic, and
// returning aggregate (struct) from device function.

// ------------------------------------------------------------------
// bool type: arithmetic, comparison result stored as bool.

__global__ void bool_arithmetic(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        bool a = (v > 0);
        bool b = (v < 100);
        bool c = a && b;
        // In C, bool used as int: true=1, false=0
        int r = (int)a + (int)b + (int)c;
        out[tid] = r;  // v in (0,100) → 3, v>100 → 1+0+0=1, v<=0 → 0+1+0=1
    }
}

// ------------------------------------------------------------------
// bool stored in array, then read back.

__global__ void bool_array(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        bool flags[4];
        flags[0] = (v & 1) != 0;
        flags[1] = (v & 2) != 0;
        flags[2] = (v & 4) != 0;
        flags[3] = (v & 8) != 0;
        int r = 0;
        for (int i = 0; i < 4; i++) {
            if (flags[i]) r += (1 << i);
        }
        out[tid] = r;  // = v & 15
    }
}

// ------------------------------------------------------------------
// printf with char* format string variants.

__global__ void printf_formats(int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && tid < n) {
        printf("tid=%d val=%d\n", tid, in[tid]);
        printf("float=%.2f\n", (double)fin[tid]);
        printf("hex=%x unsigned=%u\n", (unsigned)in[tid], (unsigned)in[tid]);
    }
}

// ------------------------------------------------------------------
// Device function with 8 parameters.

__device__ int eight_param(int a, int b, int c, int d,
                            int e, int f, int g, int h) {
    return a + b - c + d - e + f - g + h;
}

__global__ void calls_eight_param(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        out[tid] = eight_param(v, v+1, v+2, v+3, v+4, v+5, v+6, v+7);
        // v + (v+1) - (v+2) + (v+3) - (v+4) + (v+5) - (v+6) + (v+7) = 4
    }
}

// ------------------------------------------------------------------
// Output parameter: device function writes through pointer.

__device__ void compute_pair(int v, int *lo, int *hi) {
    *lo = v - 1;
    *hi = v + 1;
}

__global__ void output_param(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int lo, hi;
        compute_pair(v, &lo, &hi);
        out[tid] = lo + hi;  // (v-1)+(v+1) = 2v
    }
}

// ------------------------------------------------------------------
// unsigned char arithmetic (byte-level ops).

__global__ void uchar_arith(unsigned char *out, unsigned char *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned char v = in[tid];
        unsigned char r = (v * 3) ^ (v >> 2);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// unsigned short arithmetic.

__global__ void ushort_arith(unsigned short *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned short v = in[tid];
        unsigned short r = v * 5 + 3;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Struct returned by device function, used in global kernel.

struct Vec2 { float x, y; };

__device__ struct Vec2 rotate90(struct Vec2 v) {
    struct Vec2 r;
    r.x = -v.y;
    r.y =  v.x;
    return r;
}

__global__ void struct_return_device(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Vec2 v;
        v.x = in[tid * 2];
        v.y = in[tid * 2 + 1];
        struct Vec2 r = rotate90(v);
        out[tid * 2]     = r.x;
        out[tid * 2 + 1] = r.y;
    }
}

// ------------------------------------------------------------------
// bool as return type from device function.

__device__ bool in_range(int v, int lo, int hi) {
    return (v >= lo && v <= hi);
}

__global__ void bool_return(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        bool r = in_range(v, 10, 50);
        out[tid] = r ? 1 : 0;
    }
}
