// Probe: string/char operations, printf format variety,
// complex format strings, and edge cases in variadic handling.

// ------------------------------------------------------------------
// Printf with many format specifiers.

__global__ void printf_many(int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && n > 0) {
        printf("int=%d uint=%u float=%f double=%lf\n",
               in[0], (unsigned)in[0], fin[0], (double)fin[0]);
        printf("hex=%x oct=%o char=%c\n",
               in[0], (unsigned)in[0], (char)(in[0] & 0x7F));
    }
}

// ------------------------------------------------------------------
// Printf inside loop.

__global__ void printf_loop(int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n && i < 4; i++) {
            printf("[%d] = %d\n", i, in[i]);
        }
    }
}

// ------------------------------------------------------------------
// Printf with conditional.

__global__ void printf_cond(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (in[tid] < 0) {
            printf("tid=%d: negative value %d\n", tid, in[tid]);
        }
    }
}

// ------------------------------------------------------------------
// Printf with long long.

__global__ void printf_ll(long long *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0 && n > 0) {
        printf("ll=%lld ull=%llu\n", in[0], (unsigned long long)in[0]);
    }
}

// ------------------------------------------------------------------
// Printf with pointer (for debugging).

__global__ void printf_ptr(int *buf, int n) {
    if (threadIdx.x == 0) {
        printf("buf ptr = %p, n = %d\n", (void*)buf, n);
    }
}

// ------------------------------------------------------------------
// Multiple printf calls per thread.

__global__ void printf_multi(int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > 0) printf("pos: tid=%d v=%d\n", tid, v);
        if (v == 0) printf("zero: tid=%d\n", tid);
        if (v < 0) printf("neg: tid=%d v=%d\n", tid, v);
    }
}
