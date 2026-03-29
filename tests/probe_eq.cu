// Probe: enum types used in expressions, enum as array index
// Also: enum comparison, switch on enum value

enum Direction {
    DIR_NORTH = 0,
    DIR_EAST  = 1,
    DIR_SOUTH = 2,
    DIR_WEST  = 3,
    DIR_COUNT = 4
};

enum Status {
    STATUS_OK    = 0,
    STATUS_ERROR = 1,
    STATUS_DONE  = 2
};

__device__ float dir_offset_x(int dir) {
    float offsets[4] = {0.0f, 1.0f, 0.0f, -1.0f};
    return offsets[dir];
}

__device__ float dir_offset_y(int dir) {
    float offsets[4] = {1.0f, 0.0f, -1.0f, 0.0f};
    return offsets[dir];
}

__global__ void enum_index(float *out_x, float *out_y, int *dirs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int d = dirs[tid] % DIR_COUNT;
        out_x[tid] = dir_offset_x(d);
        out_y[tid] = dir_offset_y(d);
    }
}

__global__ void enum_switch(int *out, int *status_in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = status_in[tid];
        int result;
        switch (s) {
            case STATUS_OK:    result = 1; break;
            case STATUS_ERROR: result = -1; break;
            case STATUS_DONE:  result = 0; break;
            default:           result = -2; break;
        }
        out[tid] = result;
    }
}
