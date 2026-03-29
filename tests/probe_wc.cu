// Probe: optimizer correctness — LICM must not hoist stores, CSE must not
// merge device calls, constant dead branch elimination, zero-trip loop,
// and boundary-value constant folding.

// ------------------------------------------------------------------
// LICM safety: store inside loop must stay in loop, not be hoisted.

__global__ void licm_store_safety(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        // The store is loop-variant (writes different values per iteration)
        for (int i = 0; i < 4; i++) {
            out[tid] = v + i;   // store must stay in loop, cannot hoist
        }
        // Final value: v + 3
    }
}

// ------------------------------------------------------------------
// LICM safety: address computation is loop-invariant, but the load
// from that address should also not be hoisted if it follows a store.

__global__ void licm_addr_then_store(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int *p = out + tid;     // address is loop-invariant
        for (int i = 0; i < 4; i++) {
            *p = v * i;          // store via loop-invariant pointer, loop-variant value
        }
    }
}

// ------------------------------------------------------------------
// Two device function calls with same args — must NOT CSE (calls are opaque).

__device__ int stateful_fn(int v) {
    // Pretends to have a side effect by reading threadIdx
    return v + (int)threadIdx.x;
}

__global__ void no_cse_calls(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r1 = stateful_fn(v);   // first call
        int r2 = stateful_fn(v);   // second call — different value if threadIdx changes
        out[tid] = r1 + r2;        // should be 2*(v + threadIdx.x)
    }
}

// ------------------------------------------------------------------
// Dead branch: constant condition that's always false.

__global__ void dead_false_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r = v;
#if 0
        r = r * 999;    // dead code — never compiled in
#endif
        if (1 == 0) {   // always false constant condition
            r = -1;     // should be dead
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Dead branch: constant condition that's always true.

__global__ void dead_true_branch(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int r;
        if (1 == 1) {   // always true — else branch is dead
            r = v * 2;
        } else {
            r = v * 3;  // dead
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Zero-trip loop: for(i = 0; i < 0; i++) — body never executes.

__global__ void zero_trip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = in[tid];
        for (int i = 0; i < 0; i++) {
            acc += 999;     // never executes
        }
        out[tid] = acc;     // = in[tid] unchanged
    }
}

// ------------------------------------------------------------------
// Constant fold with signed overflow boundary (INT_MAX - related).

__global__ void const_fold_boundary(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // All compile-time constants — should fold to 0
        int a = 2147483647;    // INT_MAX
        int b = -2147483647;   // -INT_MAX
        int r = a + b;         // 0 (no overflow on these values)
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Const fold: multiple constant expressions in single statement.

__global__ void multi_const_fold(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int r = (3 * 7) + (100 / 4) - (15 % 7) + (8 << 2);
        // = 21 + 25 - 1 + 32 = 77
        out[tid] = r + tid;
    }
}

// ------------------------------------------------------------------
// Loop with compile-time trip count that is EXACTLY the unroll threshold.

__global__ void trip16_exact(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 16; i++) {
            acc += v + i;
        }
        out[tid] = acc;  // 16*v + (0+1+...+15) = 16*v + 120
    }
}

// ------------------------------------------------------------------
// Loop with compile-time trip count one above threshold (must stay a loop).

__global__ void trip17_stays_loop(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int acc = 0;
        for (int i = 0; i < 17; i++) {
            acc += v + i;
        }
        out[tid] = acc;  // 17*v + (0+1+...+16) = 17*v + 136
    }
}
