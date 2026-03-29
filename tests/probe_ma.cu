// Probe: Constant folding through multi-step chains and cross-block propagation
// - Long chain: a=1, b=a+1, c=b*2, d=c-1, e=d+a → should fold to constant
// - Conditional with constant condition → dead branch elimination
// - Loop with constant trip count but non-trivial body constant folding
// - Mixed float/int constant arithmetic

__global__ void const_chain(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = 1;
        int b = a + 1;      // 2
        int c = b * 2;      // 4
        int d = c - 1;      // 3
        int e = d + a;      // 4
        int f = e * b;      // 8
        int g = f - c;      // 4
        out[tid] = g;       // should be 4 (constant)
    }
}

// Dead branch from constant condition
__global__ void dead_branch(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = 5;
        int y = 3;
        int z;
        if (x > y) {        // always true (5 > 3)
            z = x + y;      // 8
        } else {
            z = x - y;      // dead
        }
        out[tid] = z;       // should be 8
    }
}

// Loop body with constant operands that can be folded per-iteration
__global__ void loop_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            int step = 2 * 3 + 1;   // constant 7 in every iteration
            sum += step * i;        // 0, 7, 14, 21 → sum = 42
        }
        out[0] = sum;
    }
}

// Float constant arithmetic
__global__ void float_const(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float pi_approx = 3.14159f;
        float two_pi = pi_approx * 2.0f;   // ~6.28
        float half_pi = pi_approx * 0.5f;  // ~1.57
        float result = two_pi - half_pi;   // ~4.71
        out[tid] = result;
    }
}
