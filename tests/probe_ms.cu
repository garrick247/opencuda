// Probe: struct assignment in conditional branches
// - s = fn(s) in one branch, not the other (if without else)
// - s.field accessed after the if-merge — must see correct value
// - s = fn1(s) vs s = fn2(s) in if/else branches
// - struct field postfix ++ inside an if branch

struct State { float val; int count; };

__device__ State update(State s, float x) {
    State r;
    r.val   = s.val + x;
    r.count = s.count + 1;
    return r;
}

__device__ State reset(State s) {
    State r;
    r.val   = 0.0f;
    r.count = 0;
    return r;
}

// Case 1: s = fn(s) in if-branch only, then s.field after merge
__global__ void cond_update(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        State s; s.val = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > 0.0f) {
                s = update(s, in[i]);
            }
            // s.count and s.val must be correct here regardless of branch
        }
        out[0] = s.val;
        out[1] = (float)s.count;
    }
}

// Case 2: s = fn1(s) vs s = fn2(s) in if/else branches
__global__ void branch_both(float *out, float *in, int *flags, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        State s; s.val = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            if (flags[i]) {
                s = update(s, in[i]);
            } else {
                s = reset(s);
            }
        }
        out[0] = s.val;
        out[1] = (float)s.count;
    }
}

// Case 3: s.count++ inside an if branch, then used after merge
__global__ void field_inc_in_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        State s; s.val = 0.0f; s.count = 0;
        for (int i = 0; i < n; i++) {
            if (in[i] > 0) {
                s.count++;
            }
        }
        out[0] = s.count;
    }
}
