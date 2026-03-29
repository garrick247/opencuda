// Probe: clock64/clock32 return values, __nanosleep statement,
// printf inside kernel, __trap / __brkpt as statements

__global__ void clock_use(long long *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        long long t0 = clock64();
        int sum = 0;
        for (int i = 0; i < 64; i++) sum += i;
        long long t1 = clock64();
        out[tid] = t1 - t0 + (long long)sum;
    }
}

__global__ void nanosleep_use(int *out, int *in, int ns, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __nanosleep(ns);
        out[tid] = in[tid] + 1;
    }
}

// Mixed: use clock64 as a seed for a simple hash
__global__ void clock_hash(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long seed = (unsigned long long)clock64() ^ (unsigned long long)tid;
        seed ^= seed >> 17;
        seed ^= seed << 31;
        seed ^= seed >> 8;
        out[tid] = (unsigned int)(seed & 0xFFFFFFFF);
    }
}
