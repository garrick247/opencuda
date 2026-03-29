// Regression: function pointer typedef — typedef ret_ty (*Alias)(params...);
// Without fix: _parse_typedef saw LPAREN after base type and raised
//   ParseError "expected IDENT, got LPAREN '('".
// Fix: detect LPAREN after type in _parse_typedef, parse (*alias)(params),
//   register alias → return type, and skip the parameter list.

typedef float (*UnaryFn)(float);
typedef float (*BinaryFn)(float, float);
typedef int   (*Predicate)(float, float);

__device__ float negate(float x) { return -x; }
__device__ float scale(float x, float s) { return x * s; }

__global__ void func_ptr_typedef_test(float *out, float *in, float s, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Even though we can't call through the pointer type in PTX,
        // the typedef must not break compilation.
        float v = in[tid];
        v = negate(v);
        v = scale(v, s);
        out[tid] = v;
    }
}
