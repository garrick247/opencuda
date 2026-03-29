// Probe: C-style macro that expands to declaration + statement  
// - #define that includes a type: #define INT_VAR(name) int name = 0
// - Macro used as statement that declares multiple variables

#define DECLARE_PAIR(a, b) int a = 0; int b = 0
#define FOR_RANGE(var, start, end) for (int var = start; var < end; var++)

__global__ void macro_decl(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        DECLARE_PAIR(x, y);  // expands to: int x = 0; int y = 0;
        FOR_RANGE(i, 0, 8) {
            x += i;
            y += i * 2;
        }
        out[tid] = x + y + tid;
    }
}

// Test: unary operators in unusual positions
__global__ void unary_positions(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = -(-v);    // double negation = v
        int b = ~~v;      // double bitwise NOT = v
        int c = !!v;      // double logical NOT: 0 if 0, 1 otherwise
        int d = !(!v);    // same as !!v
        out[tid] = a + b + c + d;
    }
}
