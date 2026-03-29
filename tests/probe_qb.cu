// Probe: address space stress tests — multiple smem arrays, local arrays,
// const memory, and interactions between address spaces in the same kernel.

// ------------------------------------------------------------------
// Multiple shared memory arrays in the same kernel.
// Both smem arrays must have independent base-address registers.

__global__ void two_smem_arrays(int *out, int *a, int *b, int n) {
    __shared__ int sa[32];
    __shared__ int sb[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        sa[tid] = a[tid];
        sb[tid] = b[tid];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        out[tid] = sa[tid] + sb[tid];
    }
}

// ------------------------------------------------------------------
// Three shared memory arrays: tests that all three get separate regs.

__global__ void three_smem(int *out, int *a, int *b, int *c, int n) {
    __shared__ int sa[16];
    __shared__ int sb[16];
    __shared__ int sc[16];
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        sa[tid] = a[tid];
        sb[tid] = b[tid];
        sc[tid] = c[tid];
    }
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n && i < 16; i++) {
            sum += sa[i] * sb[i] + sc[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Shared memory used for prefix sum then read back.
// Multiple sync points with same smem array.

__global__ void smem_prefix(int *out, int *data, int n) {
    __shared__ int tmp[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        tmp[tid] = data[tid];
    }
    __syncthreads();
    // Each thread reads its neighbor
    if (tid > 0 && tid < n && tid < 32) {
        tmp[tid] += tmp[tid - 1];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        out[tid] = tmp[tid];
    }
}

// ------------------------------------------------------------------
// Local array (stack): small VLA-like pattern.
// int local_buf[8] on the stack — tests .local memory.

__global__ void local_array_sort(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int buf[8];
        int m = (n < 8) ? n : 8;
        for (int i = 0; i < m; i++) buf[i] = data[i];
        // Bubble sort
        for (int i = 0; i < m - 1; i++) {
            for (int j = 0; j < m - 1 - i; j++) {
                if (buf[j] > buf[j+1]) {
                    int tmp = buf[j];
                    buf[j] = buf[j+1];
                    buf[j+1] = tmp;
                }
            }
        }
        for (int i = 0; i < m; i++) out[i] = buf[i];
    }
}

// ------------------------------------------------------------------
// Smem + local array in the same kernel.
// Both address spaces active simultaneously.

__global__ void smem_and_local(int *out, int *data, int n) {
    __shared__ int smem[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        smem[tid] = data[tid] * 2;
    }
    __syncthreads();
    if (tid == 0) {
        int local[8];
        for (int i = 0; i < 8 && i < n; i++) {
            local[i] = smem[i] + i;
        }
        int sum = 0;
        for (int i = 0; i < 8 && i < n; i++) sum += local[i];
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Const pointer (passed as const): tests that ld.global is still used
// for const pointer params (no const cache for regular params).

__global__ void const_ptr_param(int *out, const int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid] + 1;
    }
}

// ------------------------------------------------------------------
// Mixed global + shared read in same expression.
// Tests that the emitter correctly handles both address spaces.

__global__ void mixed_addr_expr(int *out, int *data, int n) {
    __shared__ int offset[1];
    int tid = threadIdx.x;
    if (tid == 0) {
        offset[0] = data[0];   // global → shared
    }
    __syncthreads();
    if (tid < n) {
        out[tid] = data[tid] + offset[0];  // global + shared
    }
}
