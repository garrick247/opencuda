// Probe: C-style string operations (parsing only, not runtime)
// - const char* parameter
// - String literal assignment
// - Char array operations
// - Boolean type (C99 _Bool or C++ bool)

__device__ int str_len(const char *s) {
    int len = 0;
    while (s[len] != 0) len++;
    return len;
}

__global__ void char_ops(int *out, char *str, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // char arithmetic
        char c = str[tid];
        int is_upper = (c >= 'A' && c <= 'Z') ? 1 : 0;
        int is_lower = (c >= 'a' && c <= 'z') ? 1 : 0;
        int val = (int)(unsigned char)c;
        out[tid] = val + is_upper * 100 + is_lower * 200;
    }
}

__global__ void bool_ops(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        bool flag_a = (a[tid] > 0);
        bool flag_b = (b[tid] > 0);
        bool both = flag_a && flag_b;
        bool either = flag_a || flag_b;
        out[tid] = (int)both * 3 + (int)either;
    }
}
