// Probe: enum values used in expressions, enum as array index,
// enum comparison, enum in switch

enum Direction { NORTH = 0, SOUTH = 1, EAST = 2, WEST = 3 };
enum Status { OK = 0, ERROR = 1, PENDING = 2 };

__global__ void enum_index(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Direction dir = (Direction)(tid % 4);
        float offsets[4];
        offsets[NORTH] = in[tid] + 1.0f;
        offsets[SOUTH] = in[tid] - 1.0f;
        offsets[EAST]  = in[tid] + 2.0f;
        offsets[WEST]  = in[tid] - 2.0f;
        out[tid] = offsets[dir];
    }
}

__global__ void enum_switch(int *out, int *dirs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Direction d = (Direction)dirs[tid];
        int result = 0;
        switch (d) {
            case NORTH: result = 10; break;
            case SOUTH: result = 20; break;
            case EAST:  result = 30; break;
            case WEST:  result = 40; break;
            default:    result = -1; break;
        }
        out[tid] = result;
    }
}

__global__ void enum_cmp(int *out, int *status, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Status s = (Status)status[tid];
        out[tid] = (s == OK) ? 1 : (s == ERROR ? -1 : 0);
    }
}
