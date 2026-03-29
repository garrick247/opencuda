// Probe: two-level device-function inlining, switch fall-through,
// continue inside short-circuit, and pointer field in struct.

// ------------------------------------------------------------------
// Two-level __device__ function inlining.
// outer() calls inner(); kernel calls outer().

__device__ int inner(int x) {
    return x * x + 1;
}

__device__ int outer(int x, int y) {
    return inner(x) + inner(y);
}

__global__ void two_level_inline(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = outer(data[tid], data[tid] + 1);
    }
}

// ------------------------------------------------------------------
// Switch with fall-through between cases.

__global__ void switch_fallthrough(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid] & 3;  // 0, 1, 2, or 3
        int r = 0;
        switch (v) {
            case 0: r += 8;  // fall through
            case 1: r += 4;  // fall through
            case 2: r += 2;  // fall through
            case 3: r += 1;
                break;
            default: r = -1; break;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// continue inside a short-circuit-guarded loop body.

__global__ void continue_in_sc_loop(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (i > 0 && data[i] == data[i - 1]) continue;  // skip duplicates
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Struct with pointer field passed to __device__ function.
// Tests that PtrTy fields in a struct are handled in inline arg loading.

struct BufRef {
    int *ptr;
    int  len;
};

__device__ int sum_buf(BufRef b, int start, int end) {
    int s = 0;
    for (int i = start; i < end && i < b.len; i++) {
        s += b.ptr[i];
    }
    return s;
}

__global__ void struct_ptr_field(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        BufRef b;
        b.ptr = data;
        b.len = n;
        out[0] = sum_buf(b, 0, n / 2) + sum_buf(b, n / 2, n);
    }
}

// ------------------------------------------------------------------
// __device__ function recursive-style depth-2 + different return types.

__device__ float squared(float x) { return x * x; }

__device__ float rms_two(float a, float b) {
    return squared(a) + squared(b);
}

__global__ void two_level_float(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = rms_two(a[tid], b[tid]);
    }
}

// ------------------------------------------------------------------
// if / else inside switch case.

__global__ void switch_with_if(int *out, int *data, int mode, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r;
        switch (mode) {
            case 0:
                r = (v > 0) ? v : -v;
                break;
            case 1:
                if (v > 100) r = 100;
                else if (v < -100) r = -100;
                else r = v;
                break;
            default:
                r = 0;
                break;
        }
        out[tid] = r;
    }
}
