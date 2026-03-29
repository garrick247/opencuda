// Probe: patterns that stress the synthesized-predicate register path.
// When an INT32 Value (not a CmpInst) is used as CondBrTerm.cond, emit.py
// synthesizes a setp.ne predicate at terminator time.  Tests here ensure
// the index chosen does not collide with later allocated predicates.
//
// Also covers:
// - do-while with && in condition
// - integer bitwise result as && LHS (non-predicate)
// - stored && result: int flag = (a && b)
// - early-exit loop with both break and continue using &&
// - for loop with multiple conditions updating induction in body

// ------------------------------------------------------------------
// Bitwise-AND result as && LHS.
// `int flag = data[tid] & 1;  if (flag && data[tid] > 0)` —
// flag is INT32 (BinInst AND, NOT a CmpInst), tests synthesis path.

__global__ void bitwise_and_lhs(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int flag = v & 1;          // BinInst result — INT32, not predicate
        // Also have a later comparison to create overlapping pred live range
        int lo = (v > -100);       // predicate via comparison
        int hi = (v < 100);        // second predicate
        int result = 0;
        if (flag && v > 0) {       // flag as && LHS — synthesis path
            result = v;
        }
        // Use lo/hi to keep their live ranges alive past the && block
        out[tid] = result + lo + hi;
    }
}

// ------------------------------------------------------------------
// Stored && result used later.
// `int ok = (v > 0 && v < 100); ... if (ok)` — INT32 && result
// used as standalone boolean later, tests synthesis in second if.

__global__ void stored_and_result(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int ok = (v > 0 && v < 100);   // land result stored in ok
        int r = 0;
        if (ok) {           // ok is INT32 (land dest), synthesis path here
            r = v;
        }
        // Second use: ok in another && with a new comparison
        if (ok && v < 50) {
            r += 10;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// do-while with && in condition.
// `do { ... } while (i < n && data[i] != 0)` — short-circuit in loop-back test.

__global__ void do_while_and(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        do {
            sum += data[i];
            i++;
        } while (i < n && data[i - 1] != 0);
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Early-exit loop: break on && and continue on another &&.
// Tests label uniqueness when multiple short-circuit blocks exist in loop body.

__global__ void break_continue_and(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            // skip negative small values
            if (data[i] < 0 && data[i] > -10) continue;
            // stop at large negative
            if (data[i] < -100 && sum > 0) break;
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Overlapping predicate live ranges with && in ternary.
// Four comparisons whose live ranges overlap, plus an && whose result
// feeds a ternary — maximizes predicate register pressure to stress
// the synthesis index assignment.

__global__ void pred_pressure_and(int *out, int *a, int *b, int *c, int *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int va = a[tid], vb = b[tid], vc = c[tid], vd = d[tid];
        // Four comparisons — potentially 4 live predicates
        int p1 = (va > 0);
        int p2 = (vb > 0);
        int p3 = (vc > 0);
        int p4 = (vd > 0);
        // && of two stored pred results (both are INT32 land/CmpInst)
        int combo = (p1 && p2) && (p3 && p4);
        // Use all four in output to keep live ranges extended
        out[tid] = combo ? (va + vb + vc + vd) : (p1 + p2 + p3 + p4);
    }
}

// ------------------------------------------------------------------
// || result stored and used as loop condition.
// `int any = (a != 0 || b != 0); while (any && i < n) { ... }`
// Tests that an INT32 lor-result feeds a while CondBrTerm correctly.

__global__ void stored_or_while(int *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int i = 0;
        // Compute initial "any" outside loop — INT32 lor result
        int any = (a[0] != 0 || b[0] != 0);
        while (any && i < n) {
            count++;
            i++;
            any = (a[i < n ? i : 0] != 0 || b[i < n ? i : 0] != 0);
        }
        out[0] = count;
    }
}
