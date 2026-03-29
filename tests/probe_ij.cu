// Probe: multiple levels of indirection, struct within struct within array,
// typedef chain resolution, function returning struct by value

typedef unsigned int uint;
typedef float float4_arr[4];

struct Matrix2x2 {
    float a, b, c, d;  // row-major: [0][0], [0][1], [1][0], [1][1]
};

struct Transform {
    Matrix2x2 rot;
    float tx, ty;
};

__device__ Matrix2x2 mat_mul(Matrix2x2 A, Matrix2x2 B) {
    Matrix2x2 C;
    C.a = A.a*B.a + A.b*B.c;
    C.b = A.a*B.b + A.b*B.d;
    C.c = A.c*B.a + A.d*B.c;
    C.d = A.c*B.b + A.d*B.d;
    return C;
}

__global__ void apply_transform(float *ox, float *oy,
                                  float *ix, float *iy,
                                  Transform *T, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float x = ix[tid];
        float y = iy[tid];
        ox[tid] = T->rot.a * x + T->rot.b * y + T->tx;
        oy[tid] = T->rot.c * x + T->rot.d * y + T->ty;
    }
}

// typedef uint usage
__global__ void uint_typedef(uint *out, uint *in, uint mask, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint v = in[tid];
        out[tid] = (v & mask) | (~v & ~mask);
    }
}
