// Regression: typedef enum { ... } Name; — typedef with enum body
// Without fix: _parse_typedef called _parse_type_with_ptr which raised
//   ParseError "expected type, got 'enum'".
// Fix: _parse_typedef checks KW_ENUM first; calls _parse_enum_def (which
//   registers enumerator values as global consts), then registers the
//   typedef alias as INT32.

typedef enum {
    MODE_NONE  = 0,
    MODE_ADD   = 1,
    MODE_MUL   = 2,
    MODE_MAX   = 3
} OpMode;

typedef enum {
    FLAG_A = 0x01,
    FLAG_B = 0x02,
    FLAG_C = 0x04
} Flags;

__global__ void enum_switch(float *out, float *a, float *b, int n, int mode) {
    int tid = threadIdx.x;
    if (tid < n) {
        switch (mode) {
            case MODE_ADD: out[tid] = a[tid] + b[tid]; break;
            case MODE_MUL: out[tid] = a[tid] * b[tid]; break;
            case MODE_MAX: out[tid] = a[tid] > b[tid] ? a[tid] : b[tid]; break;
            default: out[tid] = 0.0f; break;
        }
    }
}

__global__ void flags_kernel(int *data, int *flags_arr, int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int f = flags_arr[tid];
        int result = 0;
        if (f & FLAG_A) result += data[tid];
        if (f & FLAG_B) result += data[tid] * 2;
        if (f & FLAG_C) result += data[tid] * 4;
        out[tid] = result;
    }
}
