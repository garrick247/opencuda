// Probe: uninitialized use of local variable (parser should not crash),
// address-of local array element (&arr[i]), pointer to local struct

__global__ void addr_of_local(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float arr[4];
        arr[0] = 1.0f;
        arr[1] = 2.0f;
        arr[2] = 3.0f;
        arr[3] = 4.0f;
        // Address of array element, pass to device function implicitly via index
        float *p = &arr[tid & 3];
        out[tid] = *p;
    }
}

// Pointer to local struct — take address, pass to helper
struct Point2 {
    float x, y;
};

__device__ float point_len_sq(Point2 *p) {
    return p->x * p->x + p->y * p->y;
}

__global__ void ptr_to_local_struct(float *out, float *x_vals, float *y_vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Point2 pt;
        pt.x = x_vals[tid];
        pt.y = y_vals[tid];
        out[tid] = point_len_sq(&pt);
    }
}
