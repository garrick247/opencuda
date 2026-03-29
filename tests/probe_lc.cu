// Probe: printf with multiple types and format specifiers,
// printf inside conditional, printf inside loop,
// printf with float and int mix

// printf basic: int and string
__global__ void printf_basic(int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            printf("in[%d] = %d\n", i, in[i]);
        }
    }
}

// printf with float
__global__ void printf_float(float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            printf("val[%d] = %f\n", i, (double)in[i]);
        }
    }
}

// printf in conditional: only some threads print
__global__ void printf_cond(int *in, int n, int threshold) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > threshold) {
            printf("tid=%d val=%d exceeded threshold=%d\n", tid, v, threshold);
        }
    }
}

// printf with multiple args of different types
__global__ void printf_mixed(int *in, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = in[tid];
        float fv = fin[tid];
        printf("tid=%d int=%d float=%f sum=%f\n",
               tid, iv, (double)fv, (double)(iv + fv));
    }
}
