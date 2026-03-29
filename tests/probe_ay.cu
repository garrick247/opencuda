// Probe: string literal in non-printf context, sizeof operator,
//        __builtin_expect, multiple struct pointer members, 
//        function call result discarded (void context)

__device__ void update_stats(int *count, float *sum, float val) {
    *count += 1;
    *sum += val;
}

__global__ void running_stats(float *out_sum, int *out_count, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float total = 0.0f;
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            update_stats(&cnt, &total, in[i]);
        }
        *out_sum = total;
        *out_count = cnt;
    }
}

// sizeof used as array size
__global__ void sizeof_usage(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int buf[sizeof(float)];  // sizeof(float) = 4
        for (int i = 0; i < (int)sizeof(float); i++) {
            buf[i] = i * tid;
        }
        out[tid] = buf[0] + buf[1] + buf[2] + buf[3];
    }
}
