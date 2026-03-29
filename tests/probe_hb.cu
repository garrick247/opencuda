// Probe: const qualifier in unusual positions — east/west const,
// const pointer to const, volatile pointer, restrict combinations

__global__ void east_const(int const *in, int * const out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid] * 2;
    }
}

// Double pointer with const
__global__ void const_ptr_ptr(const float * const *matrix, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = matrix[tid][0];
    }
}

// volatile + const combination
__global__ void vol_const(volatile const int *sensor, float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = (float)sensor[tid];
    }
}

// Pointer to function-argument array (decayed to pointer)
__device__ float sum_array(const float *arr, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += arr[i];
    return s;
}

__global__ void ptr_array_arg(float *out, const float *in, int n, int chunk) {
    int tid = threadIdx.x;
    if (tid < n / chunk) {
        out[tid] = sum_array(in + tid * chunk, chunk);
    }
}
