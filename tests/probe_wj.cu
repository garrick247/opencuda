// Probe: extern __shared__ dynamic memory, isnan/isinf/isfinite,
// multiple kernels sharing a device function, kernel taking struct param,
// and memset/memcpy device patterns.

// ------------------------------------------------------------------
// extern __shared__: dynamically-sized shared memory.

__global__ void dynamic_smem(float *out, float *in, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Read neighbor from dynamic shared
    int neighbor = (tid + 1) % blockDim.x;
    if (gid < n) {
        out[gid] = smem[tid] + smem[neighbor];
    }
}

// ------------------------------------------------------------------
// Dynamic __shared__ of int type.

__global__ void dynamic_smem_int(int *out, int *in, int n) {
    extern __shared__ int ismem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    ismem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    if (gid < n) {
        out[gid] = ismem[tid] * 2;
    }
}

// ------------------------------------------------------------------
// isnan / isinf / isfinite.

__global__ void special_float_checks(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int nan_flag  = isnan(v)   ? 1 : 0;
        int inf_flag  = isinf(v)   ? 2 : 0;
        int fin_flag  = isfinite(v) ? 4 : 0;
        out[tid] = nan_flag | inf_flag | fin_flag;
    }
}

// ------------------------------------------------------------------
// __isnanf / __isinff / __isfinitef (float-specific variants).

__global__ void special_float_checks2(int *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        int r = 0;
        if (__isnanf(v)) r |= 1;
        if (__isinff(v)) r |= 2;
        if (__isfinitef(v)) r |= 4;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Two kernels sharing same __device__ function.

__device__ float apply_gain(float v, float gain) {
    return v * gain + gain * 0.1f;
}

__global__ void kernel_a(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = apply_gain(in[tid], 2.0f);
}

__global__ void kernel_b(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = apply_gain(in[tid], 0.5f);
}

// ------------------------------------------------------------------
// Kernel taking struct parameter by value.

struct Config {
    int  width;
    int  height;
    float scale;
};

__global__ void kernel_with_struct_param(int *out, int n, struct Config cfg) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = (int)((float)(tid % cfg.width) * cfg.scale);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Device memset equivalent: fill local array.

__device__ void fill_array(int *arr, int len, int val) {
    for (int i = 0; i < len; i++) arr[i] = val;
}

__global__ void device_memset(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int local[8];
        fill_array(local, 8, tid + 1);
        int sum = 0;
        for (int i = 0; i < 8; i++) sum += local[i];
        out[tid] = sum;   // 8 * (tid + 1)
    }
}

// ------------------------------------------------------------------
// Device memcpy equivalent: copy local arrays.

__device__ void copy_array(int *dst, const int *src, int len) {
    for (int i = 0; i < len; i++) dst[i] = src[i];
}

__global__ void device_memcpy(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid * 4 + 3 < n) {
        int tmp[4];
        copy_array(tmp, in + tid * 4, 4);
        out[tid] = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}
