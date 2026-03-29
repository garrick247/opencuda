// Regression: enum constants, sizeof(), and prefix ++/-- operators.
// Without fix:
//   - enum: "undefined variable 'RED'" (enum not parsed)
//   - sizeof: "unexpected token 'float'" (sizeof not parsed)
//   - ++i: "unexpected token '++'" (prefix increment not parsed)

enum Direction { NORTH = 0, EAST = 1, SOUTH = 2, WEST = 3 };
enum Flags { FLAG_A = 1, FLAG_B = 2, FLAG_C = 4, FLAG_ALL = 7 };

__global__ void enum_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int dir = (tid % 4);
        // Use enum values as constants
        int is_east = (dir == EAST) ? 1 : 0;
        int flags = FLAG_A | FLAG_C;  // = 5
        out[tid * 2 + 0] = is_east;
        out[tid * 2 + 1] = flags;
    }
}

__global__ void sizeof_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sf = sizeof(float);     // 4
        int si = sizeof(int);       // 4
        int sd = sizeof(double);    // 8
        int sll = sizeof(long long); // 8
        out[tid * 4 + 0] = sf;
        out[tid * 4 + 1] = si;
        out[tid * 4 + 2] = sd;
        out[tid * 4 + 3] = sll;
    }
}

__global__ void prefix_incdec_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = tid;
        int a = ++i;   // pre-increment: a = tid+1, i = tid+1
        int b = --i;   // pre-decrement: b = tid, i = tid
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}
