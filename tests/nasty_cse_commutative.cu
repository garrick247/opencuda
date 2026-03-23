// Tests commutative CSE: a+b and b+a in the same block should merge.
__global__ void commutative_cse(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int x = a[tid];
    int y = b[tid];
    // These four pairs are identical under commutativity — should CSE to two values.
    int add1 = x + y;
    int add2 = y + x;    // same as add1
    int mul1 = x * y;
    int mul2 = y * x;    // same as mul1
    out[tid] = add1 + add2 + mul1 + mul2;
}

__global__ void bitwise_commutative(unsigned *out, unsigned *a, unsigned *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    unsigned x = a[tid];
    unsigned y = b[tid];
    // AND, OR, XOR are also commutative — 3 redundant ops should vanish.
    unsigned and1 = x & y;
    unsigned and2 = y & x;   // redundant
    unsigned or1  = x | y;
    unsigned or2  = y | x;   // redundant
    unsigned xor1 = x ^ y;
    unsigned xor2 = y ^ x;   // redundant
    out[tid] = and1 + and2 + or1 + or2 + xor1 + xor2;
}
