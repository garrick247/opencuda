// Probe: extern __shared__ dynamic shared memory, multi-dim local arrays,
// inline device functions, and __device__ global pointer variables.

// ------------------------------------------------------------------
// extern __shared__: dynamic shared memory (size passed at launch).

__global__ void dynamic_shared(float *out, float *in, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    if (tid < n) {
        smem[tid] = in[tid] * 2.0f;
        __syncthreads();
        out[tid] = smem[(tid + 1) % n];
    }
}

// ------------------------------------------------------------------
// Extern shared with custom type.

__global__ void dynamic_shared_int(int *out, int *in, int n) {
    extern __shared__ int ismem[];
    int tid = threadIdx.x;
    if (tid < n) {
        ismem[tid] = in[tid] + tid;
        __syncthreads();
        out[tid] = ismem[(n - 1 - tid)];
    }
}

// ------------------------------------------------------------------
// Multi-dimensional local array: float A[4][4].

__global__ void matmul4x4_local(float *C, float *A, float *B) {
    float tmp[4][4];
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < 4 && col < 4) {
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            sum += A[row * 4 + k] * B[k * 4 + col];
        }
        tmp[row][col] = sum;
        C[row * 4 + col] = tmp[row][col];
    }
}

// ------------------------------------------------------------------
// __device__ inline function: compiler should inline it.

__device__ __inline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__global__ void lerp_kernel(float *out, float *a, float *b, float *t, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = lerp(a[tid], b[tid], t[tid]);
    }
}

// ------------------------------------------------------------------
// __device__ global pointer: pointer stored in device memory.

__device__ float *g_device_ptr;

__global__ void use_device_ptr(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = g_device_ptr[tid];
    }
}

// ------------------------------------------------------------------
// Shared memory with multiple types in a struct (manual layout).

struct SMLayout {
    float a[16];
    int   b[16];
};

__global__ void shared_struct_layout(float *out, float *in, int *idx, int n) {
    __shared__ SMLayout sm;
    int tid = threadIdx.x;
    if (tid < 16 && tid < n) {
        sm.a[tid] = in[tid];
        sm.b[tid] = idx[tid];
        __syncthreads();
        out[tid] = sm.a[sm.b[tid] % 16];
    }
}

// ------------------------------------------------------------------
// Pointer to shared array element: &smem[k].

__global__ void shared_ptr_elem(float *out, float *in, int n) {
    __shared__ float s[32];
    int tid = threadIdx.x;
    if (tid < 32 && tid < n) {
        s[tid] = in[tid];
        __syncthreads();
        float *p = &s[(tid + 1) % 32];
        out[tid] = *p + s[tid];
    }
}
