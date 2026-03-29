// Probe: Complex switch patterns
// - Fall-through between cases (no break)
// - Switch on enum value
// - Switch inside a loop
// - Case with complex expression as case value (constant expression)
// - Default in the middle

enum OpCode {
    OP_ADD = 0,
    OP_SUB = 1,
    OP_MUL = 2,
    OP_DIV = 3,
    OP_NOP = 4
};

__global__ void switch_fallthrough(int *out, int *ops, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid];
        int result = 0;
        switch (ops[tid]) {
            case OP_ADD:
                result = va + vb;
                break;
            case OP_SUB:
                result = va - vb;
                break;
            case OP_MUL:
                result = va * vb;
                break;
            case OP_DIV:
                result = (vb != 0) ? va / vb : 0;
                break;
            default:
                result = 0;
                break;
        }
        out[tid] = result;
    }
}

// Switch inside loop
__global__ void loop_with_switch(int *out, int *modes, int n, int iters) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = tid;
        int mode = modes[tid];
        for (int i = 0; i < iters; i++) {
            switch (mode) {
                case 0: acc += i; break;
                case 1: acc -= i; break;
                case 2: acc ^= i; break;
                default: acc |= i; break;
            }
        }
        out[tid] = acc;
    }
}
