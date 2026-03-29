// Probe: nasty register liveness patterns — variable defined in one branch
// and used after merge, loop with early exit modifying multiple vars,
// device fn returning through multiple paths affecting caller registers.

// ------------------------------------------------------------------
// Variable conditionally defined, used after if-else merge.

__global__ void cond_def_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result;
        if (v > 0) {
            result = v * 2;
        } else if (v < 0) {
            result = -v;
        } else {
            result = 1;
        }
        // result is defined on all paths — must be live here
        out[tid] = result + tid;
    }
}

// ------------------------------------------------------------------
// Loop that modifies 3 variables, early exit path vs normal exit.

__global__ void loop_multi_exit_vars(int *out_a, int *out_b, int *out_c,
                                      int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = 1, c = v;
        for (int i = 0; i < 8; i++) {
            a += c;
            b *= (c & 1) ? 2 : 1;
            c = (c >> 1) ^ (c & 1 ? 0xB4 : 0);  // LFSR-like
            if (c == 0) break;  // early exit
        }
        out_a[tid] = a;
        out_b[tid] = b;
        out_c[tid] = c;
    }
}

// ------------------------------------------------------------------
// Device fn with 3 return paths; caller uses return value in loop.

__device__ int classify3(int v, int lo, int hi) {
    if (v < lo) return -1;
    if (v > hi) return 1;
    return 0;
}

__global__ void search_range(int *out, int *data, int *los, int *his, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int cnt_below = 0, cnt_above = 0, cnt_in = 0;
        for (int i = 0; i < n; i++) {
            int c = classify3(data[i], los[tid], his[tid]);
            if (c < 0) cnt_below++;
            else if (c > 0) cnt_above++;
            else cnt_in++;
        }
        out[tid] = cnt_below * 100 + cnt_in * 10 + cnt_above;
    }
}

// ------------------------------------------------------------------
// Phi at loop header with back-edge from nested block.

__global__ void nested_loop_phi(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if ((i + j) % 2 == 0) {
                    sum += i * 4 + j;
                }
            }
        }
        out[tid] = sum + tid;
        // sum = 0+2+4+6+2+4+6+8 = but only even i+j positions
        // (0,0)=0, (0,2)=2, (1,1)=5, (1,3)=7, (2,0)=8, (2,2)=10, (3,1)=13, (3,3)=15
        // sum = 0+2+5+7+8+10+13+15 = 60
    }
}

// ------------------------------------------------------------------
// do-while(0) macro simulating CHECK with early return.

#define CHECK_RANGE(v, lo, hi, out, tid) do { \
    if ((v) < (lo) || (v) > (hi)) {           \
        (out)[tid] = -999;                     \
        return;                                \
    }                                          \
} while(0)

__global__ void checked_compute(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        CHECK_RANGE(v, 0, 1000, out, tid);
        int r = v * v;
        CHECK_RANGE(r, 0, 500000, out, tid);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Multiple variables live across a device call.

__device__ int transform(int x, int y) {
    return x * 3 + y * 7;
}

__global__ void multi_live_across_call(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid];
        int b = a + 1;
        int c = a + 2;
        int d = a + 3;
        // b, c, d must stay live across the transform call
        int t = transform(a, b);
        out[tid] = t + c + d;  // all three used after call
    }
}

// ------------------------------------------------------------------
// Shared → register → shared round-trip with predication.

__global__ void smem_roundtrip_pred(int *out, int *in, int n) {
    __shared__ int smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? in[gid] : 0;
    __syncthreads();

    // Read neighbor with predication
    int val = smem[tid];
    int nbr = (tid > 0) ? smem[tid - 1] : smem[tid];

    // Conditionally write back
    if (val > nbr) {
        smem[tid] = val - nbr;
    }
    __syncthreads();

    if (gid < n) {
        out[gid] = smem[tid];
    }
}

// ------------------------------------------------------------------
// Warp divergence: alternating even/odd threads do different work.

__global__ void warp_diverge(int *out_even, int *out_odd, int *in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (tid % 2 == 0) {
            out_even[tid / 2] = v * v;
        } else {
            out_odd[tid / 2]  = v + v;
        }
    }
}

// ------------------------------------------------------------------
// Multiple conditions updating a flag variable.

__global__ void flag_update(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int flags = 0;
        if (v & 0x01) flags |= 0x01;
        if (v & 0x02) flags |= 0x02;
        if (v & 0x04) flags |= 0x04;
        if (v & 0x08) flags |= 0x08;
        if (v > 128)  flags |= 0x10;
        if (v < 0)    flags |= 0x20;
        if (v == 0)   flags |= 0x40;
        if (__popc(v) > 4) flags |= 0x80;
        out[tid] = flags;
    }
}
