// Probe: optimizer correctness — constant folding safety, CSE boundary
// correctness, and LICM interaction with loop-carried dependencies.

// ------------------------------------------------------------------
// Constant that changes across loop iterations must NOT be folded.

__global__ void no_fold_loop(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = tid;
        for (int i = 1; i <= 8; i++) {
            acc = acc * 2 + i;  // i is NOT constant — varies per iteration
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Identical sub-expressions in different branches (CSE safe if same block).

__global__ void cse_in_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sq = v * v;       // subexpr 1
        if (v > 0) {
            int sq2 = v * v;  // same as sq — CSE should merge within branch
            out[tid] = sq + sq2;
        } else {
            out[tid] = -sq;
        }
    }
}

// ------------------------------------------------------------------
// Expression that looks invariant but depends on loop var.

__global__ void quasi_invariant(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            // (v + 1) looks like loop-invariant but should be CSE'd once
            // i * (v + 1) is NOT invariant — i changes
            acc += i * (v + 1);
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Strength reduction opportunity: sum of i*v → v * N*(N-1)/2.

__global__ void strength_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += v * i;
        }
        // sum should equal v * (0+1+2+...+7) = v * 28
        // but we just test correctness, not whether it was reduced
        out[tid] = sum;
    }
}

// ------------------------------------------------------------------
// Double-computation: same expr used twice in different contexts.

__global__ void double_use(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int av = a[tid], bv = b[tid];
        int diff = av - bv;
        int sq_diff = diff * diff;  // used twice
        out[tid * 2 + 0] = sq_diff;
        out[tid * 2 + 1] = sq_diff + diff;
    }
}

// ------------------------------------------------------------------
// Constant folding: chain of const-only arithmetic at block entry.

__global__ void const_chain(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // These should be folded away at compile time
        int k1 = 2 * 3 + 4;      // = 10
        int k2 = k1 * k1 - 5;   // = 95
        int k3 = (k2 + 5) / 2;  // = 50
        out[tid] = v * k3;
    }
}

// ------------------------------------------------------------------
// Loop invariant load: same global address loaded in every iteration.

__global__ void loop_inv_load(int *out, int *in, int *scale, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = scale[0];  // ideally hoisted out of loop, but correctness is the goal
        int acc = 0;
        for (int i = 0; i < 8; i++) {
            acc += in[tid] * s + i;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Unroll target: small fixed-count loop with known trip count.

__global__ void unroll_target(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = 0;
        // Trip count = 4 ≤ 16 — unroller may unroll this
        for (int i = 0; i < 4; i++) {
            r = r * 2 + v + i;
        }
        out[tid] = r;
    }
}
