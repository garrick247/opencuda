// Probe: Unusual initializer and declaration patterns
// - int x = (int)expr without a separate variable
// - Array declared with sizeof: int arr[sizeof(float) * N]  
// - Multiple qualifiers: const unsigned int
// - Signed keyword: signed int, signed char
// - Array size from constant expression

#define N 32

__constant__ int c_lut[N];

__global__ void lut_lookup(int *out, int *in, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        int idx = in[tid] % N;
        out[tid] = c_lut[idx];
    }
}

__global__ void signed_types(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        signed int sv = in[tid];
        signed char sc = (signed char)(sv & 0x7F);
        const unsigned int uv = (unsigned int)sv;
        out[tid] = (int)sc + (int)uv;
    }
}

__global__ void multi_qualifier(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const int kStride = 4;
        int sum = 0;
        for (int i = tid; i < n; i += kStride) {
            sum += i;
        }
        out[tid] = sum;
    }
}
