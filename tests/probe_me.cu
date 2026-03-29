// Probe: Nested conditional + phi merging edge cases
// - Triple-nested if/else: value defined in all 8 leaf branches
// - Early return inside nested if (value may be undefined on other paths)
// - Phi of a phi: outer merge depends on inner merge result
// - Conditional with assignment in only one branch (value uninitialized on other path)
// - Ternary nested inside ternary

__global__ void triple_nested(int *out, int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = a[tid];
        int y = b[tid];
        int z = c[tid];
        int result;
        if (x > 0) {
            if (y > 0) {
                if (z > 0) {
                    result = x + y + z;
                } else {
                    result = x + y - z;
                }
            } else {
                if (z > 0) {
                    result = x - y + z;
                } else {
                    result = x - y - z;
                }
            }
        } else {
            if (y > 0) {
                if (z > 0) {
                    result = -x + y + z;
                } else {
                    result = -x + y - z;
                }
            } else {
                if (z > 0) {
                    result = -x - y + z;
                } else {
                    result = -x - y - z;
                }
            }
        }
        out[tid] = result;
    }
}

// phi of phi: inner if produces val, outer if uses inner result
__global__ void phi_of_phi(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int inner;
        if (v > 10) {
            inner = v * 2;
        } else {
            inner = v + 1;
        }
        // inner is now a phi result
        int outer;
        if (inner > 15) {
            outer = inner - 5;
        } else {
            outer = inner + 5;
        }
        // outer depends on inner which is a phi
        out[tid] = outer;
    }
}

// Assignment only in one branch — other path uses uninitialized (will be zero-init)
__global__ void one_branch_assign(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int extra;
        if (v > 100) {
            extra = v * 3;
        }
        // extra is undefined if v <= 100 — zero-init kicks in
        // result is v + extra (may be v+0 or v + v*3)
        out[tid] = v + extra;
    }
}

// Nested ternary
__global__ void nested_ternary(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int result = (v > 0) ? ((v > 10) ? v * 3 : v * 2) : -v;
        out[tid] = result;
    }
}

// Conditional loop exit + phi after loop
__global__ void cond_loop_exit(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int sum = 0;
        int found = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            if (sum > 1000) {
                found = i;
                break;
            }
        }
        // found is 0 if never broke, or the break index
        out[tid] = found;
    }
}
