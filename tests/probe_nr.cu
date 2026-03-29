// Probe: continue-guarded inline, break-after-inline early exit,
//        inter-field dependency inside inline (second = best before best = v)

// ------------------------------------------------------------------
// Continue-guarded inline: negative values skip the struct update.
// Tests that struct fields remain live across the bypass path
// (for_inc_7 → for_cond_5) that does NOT pass through inline_merge.

struct Pos2 { float sum; float sumsq; };

__device__ Pos2 accum_pos(Pos2 a, float v) {
    a.sum   += v;
    a.sumsq += v * v;
    return a;
}

__global__ void continue_guard(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Pos2 a; a.sum = 0.0f; a.sumsq = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v < 0.0f) continue;
            a = accum_pos(a, v);
        }
        out[0] = a.sum;
        out[1] = a.sumsq;
    }
}

// ------------------------------------------------------------------
// Break after inline: loop exits early when the running total exceeds
// a threshold.  Tests that both struct fields are live on the break
// path to for_exit (they must be stored to output).

struct RunSum { float val; int cnt; };

__device__ RunSum rs_add(RunSum s, float v) {
    s.val += v;
    s.cnt += 1;
    return s;
}

__global__ void break_after_inline(float *out, float *data, int n, float limit) {
    int tid = threadIdx.x;
    if (tid == 0) {
        RunSum s; s.val = 0.0f; s.cnt = 0;
        for (int i = 0; i < n; i++) {
            s = rs_add(s, data[i]);
            if (s.val > limit) break;
        }
        out[0] = s.val;
        out[1] = (float)s.cnt;
    }
}

// ------------------------------------------------------------------
// Inter-field dependency inside inline: s.second = s.best (old) is
// assigned BEFORE s.best = v.  Both assignments are in the same
// if-block body.  Tests that the parser preserves assignment order:
// second must hold the OLD best, not the new v.

struct Best3 { float best; float second; float worst; };

__device__ Best3 rank_update(Best3 s, float v) {
    if (v > s.best) {
        s.second = s.best;
        s.best   = v;
    }
    if (v < s.worst) {
        s.worst = v;
    }
    return s;
}

__global__ void rank_tracker(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Best3 s;
        s.best   = data[0];
        s.second = data[0];
        s.worst  = data[0];
        for (int i = 1; i < n; i++) {
            s = rank_update(s, data[i]);
        }
        out[0] = s.best;
        out[1] = s.second;
        out[2] = s.worst;
    }
}
