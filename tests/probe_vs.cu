// Probe: __ldg on various types, volatile __shared__, 2D shared memory,
// and designated struct initializers.

// ------------------------------------------------------------------
// __ldg on various scalar types.

__global__ void ldg_types(int *iout, float *fout, double *dout,
                           int *iin, float *fin, double *din, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int   iv = __ldg(iin + tid);
        float fv = __ldg(fin + tid);
        double dv = __ldg(din + tid);
        iout[tid] = iv + 1;
        fout[tid] = fv * 2.0f;
        dout[tid] = dv + 1.0;
    }
}

// ------------------------------------------------------------------
// __ldg on struct (float2).

__global__ void ldg_float2(float2 *out, float2 *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float2 v = __ldg(in + tid);
        float2 r;
        r.x = v.x + v.y;
        r.y = v.x * v.y;
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Volatile __shared__ — all accesses must not be optimized away.

__global__ void volatile_shared(float *out, float *in, int n) {
    __shared__ volatile float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();

    // Adjacent pair sum
    if (gid < n && tid > 0) {
        out[gid] = smem[tid] + smem[tid - 1];
    } else if (gid < n) {
        out[gid] = smem[tid];
    }
}

// ------------------------------------------------------------------
// 2D shared memory array.

__global__ void shared_2d(float *out, float *in, int rows, int cols) {
    __shared__ float smem[16][16];
    int r = threadIdx.y;
    int c = threadIdx.x;
    int gr = blockIdx.y * blockDim.y + r;
    int gc = blockIdx.x * blockDim.x + c;

    smem[r][c] = (gr < rows && gc < cols) ? in[gr * cols + gc] : 0.0f;
    __syncthreads();

    // Each thread reads its transposed neighbor
    if (gr < rows && gc < cols) {
        out[gr * cols + gc] = smem[c][r];
    }
}

// ------------------------------------------------------------------
// Array of structs in shared memory.

struct Packed {
    float val;
    int   tag;
};

__global__ void shared_packed_arr(float *out, float *in, int *tags_in, int n) {
    __shared__ struct Packed spack[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) {
        spack[tid].val = in[gid];
        spack[tid].tag = tags_in[gid];
    }
    __syncthreads();

    if (gid < n) {
        int neighbor = (tid + 1) & (blockDim.x - 1);
        float result = spack[tid].val;
        if (spack[neighbor].tag == spack[tid].tag) {
            result += spack[neighbor].val;
        }
        out[gid] = result;
    }
}

// ------------------------------------------------------------------
// sizeof of variable (not type) — sizeof(v) = sizeof(type of v).

__global__ void sizeof_var(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = tid;
        float f = (float)tid;
        double d = (double)tid;
        // sizeof(variable) should give same as sizeof(type)
        int si = sizeof(v);    // 4
        int sf = sizeof(f);    // 4
        int sd = sizeof(d);    // 8
        out[tid] = si + sf + sd;  // 16
    }
}
