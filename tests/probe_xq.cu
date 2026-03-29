// Probe: post-increment indexing, multi-assign chain, switch fallthrough,
// nested ternary, volatile shared, size_t param, complex loop conditions.

// ------------------------------------------------------------------
// Post-increment in array access: arr[i++] — side-effect ordering.

__global__ void post_inc_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n - 3) {
        int i = tid;
        // Each in[i++] should advance i by 1 after use
        int a = in[i++];   // uses i=tid,   i becomes tid+1
        int b = in[i++];   // uses i=tid+1, i becomes tid+2
        int c = in[i++];   // uses i=tid+2, i becomes tid+3
        out[tid] = a + b + c;
    }
}

// ------------------------------------------------------------------
// Multi-assignment chain: a = b = c = expr.

__global__ void multi_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a, b, c;
        c = b = a = in[tid] * 2;
        out[tid] = a + b + c;   // should be 6 * in[tid]
    }
}

// ------------------------------------------------------------------
// Switch with deliberate fallthrough (no break between cases).

__device__ int grade_score(int score) {
    int result;
    switch (score / 10) {
        case 10:
        case 9:  result = 4; break;   // A
        case 8:  result = 3; break;   // B
        case 7:  result = 2; break;   // C
        case 6:  result = 1; break;   // D
        default: result = 0; break;   // F
    }
    return result;
}

__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = grade_score(in[tid]);
    }
}

// ------------------------------------------------------------------
// Three-level nested ternary.

__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // sign3: +1, 0, -1
        int s = v > 0 ? 1 : (v < 0 ? -1 : 0);
        // clamp3: -1, v (if |v|≤100), 100
        int c = v < -1 ? -1 : (v > 100 ? 100 : v);
        out[tid] = s * c;
    }
}

// ------------------------------------------------------------------
// Volatile shared memory: prevents CSE/hoisting of shared reads.

__global__ void volatile_shared(int *out, int *in, int n) {
    volatile __shared__ int smem[256];
    int tid = threadIdx.x;
    smem[tid] = (tid < n) ? in[tid] : 0;
    __syncthreads();

    if (tid < n) {
        // Volatile reads — optimizer must not merge these
        int a = smem[tid];
        int b = smem[(tid + 1) % blockDim.x];
        out[tid] = a - b;
    }
}

// ------------------------------------------------------------------
// size_t kernel parameter.

__global__ void size_t_kernel(float *out, float *in, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = in[gid] * 2.0f;
    }
}

// ------------------------------------------------------------------
// Complex loop condition: while with && and embedded call.

__device__ int next_prime(int x) {
    // Naive next prime after x
    for (int p = x + 1; ; p++) {
        int ok = 1;
        for (int i = 2; i * i <= p && ok; i++) {
            if (p % i == 0) ok = 0;
        }
        if (ok) return p;
    }
}

__global__ void complex_loop_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = next_prime(in[tid] % 20 + 2);
    }
}

// ------------------------------------------------------------------
// Pre-decrement and mixed increment operators.

__global__ void mixed_increments(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int a = ++x;   // pre-increment: a = x+1
        int b = x--;   // post-decrement: b = x+1, then x = x
        int c = --x;   // pre-decrement: c = x-1
        out[tid] = a + b + c;  // (x+1) + (x+1) + (x-1) = 3x+1 where x=in[tid]
    }
}

// ------------------------------------------------------------------
// Compound assignment with complex RHS.

__global__ void compound_complex(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = a[tid];
        v += b[tid] * 3;
        v -= (v >> 2);
        v &= ~0xFF;
        v |= tid & 0xFF;
        out[tid] = v;
    }
}
