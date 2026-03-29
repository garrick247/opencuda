// Regression: multi-dimensional array fields in structs (float m[3][3])
// Without fix: _parse_struct_def only consumed one '[N]' for array fields;
//   second '[3]' was left unconsumed → ParseError "expected SEMI, got LBRACKET '['".
// Fix: loop while LBRACKET to consume all dimension brackets, multiplying into total count.

struct Matrix3x3 {
    float m[3][3];
};

struct Image16x16 {
    float pixels[16][16];
    int   labels[16];
};

__device__ float mat_trace(struct Matrix3x3 *mat) {
    return mat->m[0][0] + mat->m[1][1] + mat->m[2][2];
}

__global__ void matrix_trace_test(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Matrix3x3 m;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                m.m[i][j] = data[tid * 9 + i * 3 + j];
        out[tid] = mat_trace(&m);
    }
}
