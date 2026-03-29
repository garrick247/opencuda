// Probe: switch with fallthrough, const local variables, static __device__
// globals, complex while conditions, bitfield-like patterns, multi-level
// pointer dereference with offset arithmetic.

// ------------------------------------------------------------------
// switch with fallthrough (no break between cases).

__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 4;
        int result = 0;
        switch (v) {
            case 0: result += 8;  // fallthrough
            case 1: result += 4;  // fallthrough
            case 2: result += 2;  // fallthrough
            case 3: result += 1;
                break;
        }
        out[tid] = result;
        // v=0 → 8+4+2+1=15, v=1 → 7, v=2 → 3, v=3 → 1
    }
}

// ------------------------------------------------------------------
// switch with mixed break and fallthrough.

__global__ void switch_mixed(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        switch (v % 6) {
            case 0: r = 100; break;
            case 1:
            case 2: r = 200; break;  // cases 1 and 2 share code
            case 3: r = 300; break;
            case 4:
            case 5: r = 400; break;
            default: r = -1;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// const local variable.

__global__ void const_local(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        const int factor = 7;
        const float pi_approx = 3.14159f;
        int v = in[tid];
        int r = v * factor;
        float f = (float)v * pi_approx;
        out[tid] = r + (int)f;
    }
}

// ------------------------------------------------------------------
// static __device__ global variable (shared across all threads).

static __device__ int d_counter = 42;

__global__ void uses_device_static(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = d_counter + tid;  // 42 + tid
    }
}

// ------------------------------------------------------------------
// Complex while: multiple conditions with &&/|| and side-effecty updates.

__global__ void while_complex_cond(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int a = 0, b = v;
        while (a < 8 && b > 0) {
            a++;
            b -= a;
        }
        out[tid] = a;
    }
}

// ------------------------------------------------------------------
// Bitfield-like packing using shifts and masks.

__global__ void bit_pack_unpack(unsigned int *out, unsigned int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int v = in[tid];
        // Pack 4 bytes into one word (byte reversal)
        unsigned int b0 = (v >> 0)  & 0xFF;
        unsigned int b1 = (v >> 8)  & 0xFF;
        unsigned int b2 = (v >> 16) & 0xFF;
        unsigned int b3 = (v >> 24) & 0xFF;
        unsigned int packed = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        out[tid] = packed;
    }
}

// ------------------------------------------------------------------
// Pointer chain: p[i] where p points into the middle of an array.

__global__ void ptr_middle_access(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < 4 && tid < n) {
        // Point to middle of array
        int *mid = in + 2;
        out[tid] = mid[tid] + mid[-1] + mid[-2];
        // mid[-2] = in[0], mid[-1] = in[1], mid[0..3] = in[2..5]
    }
}

// ------------------------------------------------------------------
// Negative array index via pointer offset.

__global__ void ptr_neg_index(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int *p = in + tid + 1;   // points one past current element
        // Safe only when tid+1 < n, but this tests codegen not bounds
        int cur  = *(p - 1);     // in[tid]
        out[tid] = cur * 2;
    }
}

// ------------------------------------------------------------------
// While loop that accumulates with multiple exits.

__global__ void while_multi_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0;
        int i = 0;
        while (1) {            // infinite loop with internal exits
            if (i >= 10) break;
            if (i == v) break; // exit when counter matches input
            sum += i;
            i++;
        }
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Nested switch inside a loop.

__global__ void switch_in_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            switch ((v + i) % 3) {
                case 0: acc += 10; break;
                case 1: acc += 20; break;
                case 2: acc += 30; break;
            }
        }
        out[tid] = acc;
    }
}
