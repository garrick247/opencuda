// Regression: struct local variables, typedef struct, char/short types.
// Without fix:
//   struct Pair p;     → ParseError "unexpected token 'struct'"
//   typedef struct { } Vec2; → ParseError "expected IDENT, got LBRACE"
//   unsigned short v;  → ParseError "expected IDENT, got KW_SHORT"
//   char c;            → ParseError "expected type, got 'char'"

struct Pair { float a; float b; };

typedef struct { float x; float y; } Vec2;

typedef struct Point3 { float x; float y; float z; } Point3;

__global__ void struct_local_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Pair p;
        p.a = in[tid * 2 + 0];
        p.b = in[tid * 2 + 1];
        out[tid] = p.a + p.b;
    }
}

__global__ void typedef_struct_test(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 v;
        v.x = in[tid * 2 + 0];
        v.y = in[tid * 2 + 1];
        out[tid] = v.x + v.y;
    }
}

__global__ void short_char_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        short s = (short)in[tid];
        unsigned short us = (unsigned short)(s + 1);
        char c = (char)(us & 0x7f);
        unsigned char uc = (unsigned char)(c + 1);
        out[tid] = (int)s + (int)us + (int)c + (int)uc;
    }
}
