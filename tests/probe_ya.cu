// Probe: LICM stress (invariant in nested loop), CSE with commutative ops,
// loop with early exit + post-loop use, do-while with complex condition,
// switch with fallthrough + break mix, multi-return paths, pointer arithmetic
// in loop, 3-level nested loop, phi-node stress from multi-way branch,
// and __nv_bfloat16 / bf16 parse tolerance.

// ------------------------------------------------------------------
// LICM stress: loop-invariant mul in inner loop.
// The mul tid*stride should be hoisted; inner loop only varies by j.

__global__ void licm_stress(float *out, float *in, int stride, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        int base = tid * stride;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < stride; j++) {
                s += in[base + i * stride + j];
            }
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Loop with early exit (break) + post-loop use of loop variable.

__global__ void early_exit_use(int *out_val, int *out_idx,
                                  int *in, int target, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int found = -1;
        int v = 0;
        for (int i = 0; i < n; i++) {
            v = in[i];
            if (v == target) { found = i; break; }
        }
        out_val[tid] = v;
        out_idx[tid] = found;
    }
}

// ------------------------------------------------------------------
// do-while with complex (multi-term) condition.

__global__ void do_while_complex(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int steps = 0;
        do {
            v = (v % 2 == 0) ? v / 2 : 3 * v + 1;
            steps++;
        } while (v != 1 && steps < 64);
        out[tid] = steps;
    }
}

// ------------------------------------------------------------------
// switch with fallthrough into next case, then break.

__device__ int switch_fall(int x) {
    int r = 0;
    switch (x & 3) {
        case 0: r += 10;  // fall through
        case 1: r += 1; break;
        case 2: r += 20; break;
        case 3: r += 30; break;
        default: r = -1; break;
    }
    return r;
}

__global__ void switch_fall_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = switch_fall(in[tid]);
}

// ------------------------------------------------------------------
// Multi-return: returns at three different code paths.

__device__ int classify3(int v) {
    if (v < 0)  return -1;
    if (v == 0) return 0;
    return 1;
}

__global__ void classify3_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = classify3(in[tid]);
}

// ------------------------------------------------------------------
// Pointer arithmetic in loop (stride-2 gather).

__global__ void stride2_gather(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float s = 0.0f;
        float *p = in + tid;
        for (int i = 0; i < 4; i++) {
            s += *p;
            p += 2;
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// 3-level nested loop: cube sum.

__global__ void cube_sum(int *out, int *in, int dim, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = 0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    s += in[i * dim * dim + j * dim + k];
                }
            }
        }
        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Phi-node stress: 4-way if-else chain (cascaded else if).

__device__ int classify4way(int v) {
    if      (v < -100) return -3;
    else if (v <    0) return -1;
    else if (v <  100) return  1;
    else               return  3;
}

__global__ void classify4_kernel(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = classify4way(in[tid]);
}

// ------------------------------------------------------------------
// CSE with reused subexpression in different branches.

__global__ void cse_branch(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        int z = c[tid];
        int ab = x + y;   // CSE candidate
        int bc = y + z;   // CSE candidate
        // Use both in both branches
        int r;
        if (x > 0) {
            r = ab * bc + ab;
        } else {
            r = ab + bc * bc;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Modulo + conditional update pattern (FizzBuzz-style).

__global__ void fizzbuzz(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = tid + 1;
        int r = v;
        if (v % 15 == 0) r = 0;
        else if (v % 3  == 0) r = -3;
        else if (v % 5  == 0) r = -5;
        out[tid] = r;
    }
}
