// Probe: switch statement with fallthrough, default in middle,
// case with no break (explicit fallthrough)

__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid] % 4;
        int result = 0;
        switch (v) {
            case 0:
                result += 10;
                // fallthrough
            case 1:
                result += 1;
                break;
            case 2:
                result += 20;
                break;
            default:
                result += 99;
                break;
        }
        out[tid] = result;
    }
}

// Switch with string of fallthrough cases (like bit-flags)
__global__ void switch_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int flags = 0;
        switch (v) {
            case 7:
                flags |= 4;
            case 6:
            case 5:
                flags |= 2;
            case 4:
                flags |= 1;
                break;
            default:
                flags = 0;
        }
        out[tid] = flags;
    }
}
