// Probe: small loop with break (trip≤16 but break prevents unroll),
// 3-field struct with all fields accessed,
// device function with output-pointer parameter (side-effects via pointer),
// loop whose small body contains if-else (should unroll if trip≤16)

// Small loop with break: trip would be ≤ 8 but break makes count non-static
__global__ void small_loop_break(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found = -1;
        for (int i = 0; i < 8; i++) {   // trip=8 ≤ 16, but break inside
            if (in[i] < 0) {
                found = i;
                break;
            }
        }
        *out = found;
    }
}

// Small trip-count loop WITHOUT break: should unroll
__global__ void small_loop_unroll(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {   // trip=4, no break — should unroll
            sum += i * i;               // 0 + 1 + 4 + 9 = 14
        }
        *out = sum;
    }
}

// 3-field struct
struct RGB {
    int r;
    int g;
    int b;
};

__global__ void rgb_luma(int *out, struct RGB *pixels, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct RGB px = pixels[tid];
        // Approximate luma: (2*r + 4*g + b) / 7
        out[tid] = (2 * px.r + 4 * px.g + px.b) / 7;
    }
}

// Device function with output pointer: computes min and max
__device__ void minmax(int *arr, int len, int *out_min, int *out_max) {
    int mn = arr[0];
    int mx = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
    }
    *out_min = mn;
    *out_max = mx;
}

__global__ void find_minmax(int *out_min, int *out_max, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        minmax(in, n, out_min, out_max);
    }
}
