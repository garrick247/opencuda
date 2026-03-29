// Probe: type qualifiers and storage class combinations
// - const int local variable
// - volatile (should parse, treat as regular)
// - unsigned/signed without explicit base type (unsigned = unsigned int)
// - long (= int or long long?)
// - short
// - restrict pointer param

__global__ void const_local(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const int step = 4;
        int sum = 0;
        for (int i = 0; i < step; i++) {
            sum += out[(tid + i) % n];
        }
        out[tid] = sum;
    }
}

__global__ void unsigned_short_types(unsigned int *out, int n) {
    unsigned int tid = (unsigned int)threadIdx.x;
    if ((int)tid < n) {
        unsigned int a = 0xFFFFFFFFu;
        unsigned int b = tid & a;
        short s = (short)tid;
        out[tid] = b + (unsigned int)s;
    }
}

__global__ void volatile_mem(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        volatile int v = tid * 2;
        out[tid] = v + 1;
    }
}

__device__ float dot_restrict(const float * __restrict__ a,
                               const float * __restrict__ b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
