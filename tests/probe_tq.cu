// Probe: more enum patterns, anonymous struct/union fields,
// and C99-style designated initializers (if supported).

enum Direction { NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3 };
enum Color { RED = 0xFF0000, GREEN = 0x00FF00, BLUE = 0x0000FF };
enum Status { OK = 0, ERR_NULL = -1, ERR_OOB = -2, ERR_TYPE = -3 };

// ------------------------------------------------------------------
// Enum as switch value.

__global__ void dir_kernel(int *out, int *dirs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int d = dirs[tid] & 3;
        int dx = 0, dy = 0;
        switch (d) {
            case NORTH: dy =  1; break;
            case SOUTH: dy = -1; break;
            case EAST:  dx =  1; break;
            case WEST:  dx = -1; break;
        }
        out[tid * 2 + 0] = dx;
        out[tid * 2 + 1] = dy;
    }
}

// ------------------------------------------------------------------
// Enum comparison.

__global__ void status_check(int *out, int *codes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int code = codes[tid];
        if (code == OK) {
            out[tid] = 1;
        } else if (code == ERR_NULL || code == ERR_OOB) {
            out[tid] = 0;
        } else {
            out[tid] = -1;
        }
    }
}

// ------------------------------------------------------------------
// Enum value in arithmetic.

__global__ void color_pack(int *out, int *colors, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int c = colors[tid] % 3;
        int rgb;
        if (c == 0) rgb = RED;
        else if (c == 1) rgb = GREEN;
        else rgb = BLUE;
        out[tid] = rgb;
    }
}

// ------------------------------------------------------------------
// Enum as function parameter.

__device__ int dir_delta(enum Direction d, int axis) {
    if (axis == 0) {  // x-axis
        if (d == EAST)  return  1;
        if (d == WEST)  return -1;
        return 0;
    } else {  // y-axis
        if (d == NORTH) return  1;
        if (d == SOUTH) return -1;
        return 0;
    }
}

__global__ void dir_move(int *ox, int *oy, int *dirs, int *ix, int *iy, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        enum Direction d = (enum Direction)(dirs[tid] & 3);
        ox[tid] = ix[tid] + dir_delta(d, 0);
        oy[tid] = iy[tid] + dir_delta(d, 1);
    }
}

// ------------------------------------------------------------------
// Multiple enum types in same kernel.

__global__ void multi_enum(int *out, int *dirs, int *statuses, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        enum Direction d = (enum Direction)(dirs[tid] & 3);
        enum Status s = (enum Status)statuses[tid];
        int r = 0;
        if (s == OK) {
            r = (int)d * 10;
        } else {
            r = (int)s;
        }
        out[tid] = r;
    }
}
