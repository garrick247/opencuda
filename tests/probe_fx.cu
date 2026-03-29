// Probe: complex #define macros with ## token pasting and stringification
// Also: variadic-like macros (VA_ARGS style)
// Also: multi-line macros with backslash continuation

#define SWAP(T, a, b) do { T _tmp = (a); (a) = (b); (b) = _tmp; } while(0)

#define ROUND_UP(x, n)  (((x) + (n) - 1) / (n) * (n))
#define ROUND_DOWN(x, n) ((x) / (n) * (n))
#define IS_POW2(x) (((x) & ((x) - 1)) == 0)
#define NEXT_POW2(x) \
    ( (x) <= 1 ? 1 : \
      (x) <= 2 ? 2 : \
      (x) <= 4 ? 4 : \
      (x) <= 8 ? 8 : \
      (x) <= 16 ? 16 : \
      (x) <= 32 ? 32 : 64 )

__global__ void macro_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = ROUND_UP(v, 16);
        int d = ROUND_DOWN(v, 8);
        int p = IS_POW2(v) ? 1 : 0;
        int np = NEXT_POW2(v & 63);
        out[tid] = r + d + p + np;
    }
}

// SWAP with int
__global__ void swap_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = in[(tid + 1) % n];
        if (a > b) {
            SWAP(int, a, b);
        }
        out[tid] = a;
        out[(tid + 1) % n] = b;
    }
}
