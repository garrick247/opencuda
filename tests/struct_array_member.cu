// Regression: inline array members in struct/typedef struct.
// Without fix: `float data[4];` inside a struct body → ParseError
//   "expected SEMI, got LBRACKET '['".
// Fix: _parse_struct_def consumes [N] after field name, expands to N scalar
//   sub-fields (data_0, data_1, ...) for correct layout offsets.
//   _parse_postfix_expr handles v.data[i] with constant i → v_data_i.

typedef struct {
    float data[4];
    int len;
} Vec4;

typedef struct {
    int index[8];
    float weight;
} Weighted;

__global__ void vec4_ops(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec4 v;
        v.len = 4;
        v.data[0] = in[tid * 4 + 0];
        v.data[1] = in[tid * 4 + 1];
        v.data[2] = in[tid * 4 + 2];
        v.data[3] = in[tid * 4 + 3];
        float sum = v.data[0] + v.data[1] + v.data[2] + v.data[3];
        out[tid] = sum / (float)v.len;
    }
}

__global__ void weighted_gather(float *out, float *vals, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Weighted w;
        w.weight = 0.125f;
        w.index[0] = tid;
        w.index[1] = tid + 1;
        float r = vals[w.index[0]] * w.weight + vals[w.index[1]] * w.weight;
        out[tid] = r;
    }
}
