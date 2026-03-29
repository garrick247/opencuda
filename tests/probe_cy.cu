// Probe: Tricky variable scoping in blocks
// - Variables declared inside compound statements (block scope)
// - Variables in both branches of if/else
// - Variable used before and after a conditional assignment
// - Loop variable escaping loop scope (not C99 for-loop scoping issue)

__global__ void block_scope(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int base = in[tid];
        {
            // Inner block
            int tmp = base * 2;
            base = tmp + 1;  // modifies outer 'base'
        }
        out[tid] = base;
    }
}

__global__ void nested_blocks(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        if (v > 0) {
            {
                int scaled = v * 3;
                out[tid] = scaled;
            }
        } else {
            {
                int negated = -v;
                out[tid] = negated;
            }
        }
    }
}

// Variable declared before if, used after
__global__ void pre_post_if(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = 0;  // pre-if
        if (v > 0) {
            result = v;
        } else if (v < 0) {
            result = -v;
        }
        // result used post-if (has value from whichever branch ran)
        out[tid] = result + tid;
    }
}
