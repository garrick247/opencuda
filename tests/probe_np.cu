// Probe: 4-field struct with two conditional if-block updates inside inline,
//        conditional inline bypass (struct unchanged when condition false),
//        two different struct types accumulated in parallel

// ------------------------------------------------------------------
// Running stats: 4-field struct, inline has two if-blocks (min/max clamp).
// Tests that all four fields are correctly live and written back, and that
// the conditional if-blocks inside the inline don't clobber each other.

struct Stats4 { float sum; float sumsq; float mn; float mx; };

__device__ Stats4 add_sample(Stats4 s, float v) {
    s.sum   += v;
    s.sumsq += v * v;
    if (v < s.mn) s.mn = v;
    if (v > s.mx) s.mx = v;
    return s;
}

__global__ void running_stats(float *out, float *data, int n,
                                float init_mn, float init_mx) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Stats4 s;
        s.sum = 0.0f; s.sumsq = 0.0f;
        s.mn = init_mn; s.mx = init_mx;
        for (int i = 0; i < n; i++) {
            s = add_sample(s, data[i]);
        }
        out[0] = s.sum;
        out[1] = s.sumsq;
        out[2] = s.mn;
        out[3] = s.mx;
    }
}

// ------------------------------------------------------------------
// Conditional inline bypass: the inline is called ONLY if data[i] > 0.
// When the condition is false, the struct fields keep their previous values.
// Tests that the struct remains live on the bypass path and its register
// is not clobbered by the false branch.

struct Acc3 { float pos_sum; float neg_sum; int pos_cnt; };

__device__ Acc3 add_pos(Acc3 a, float v) {
    a.pos_sum += v;
    a.pos_cnt += 1;
    return a;
}

__device__ Acc3 add_neg(Acc3 a, float v) {
    a.neg_sum += v;
    return a;
}

__global__ void cond_inline_bypass(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Acc3 a; a.pos_sum = 0.0f; a.neg_sum = 0.0f; a.pos_cnt = 0;
        for (int i = 0; i < n; i++) {
            float v = data[i];
            if (v > 0.0f) {
                a = add_pos(a, v);
            } else {
                a = add_neg(a, v);
            }
        }
        out[0] = a.pos_sum;
        out[1] = a.neg_sum;
        out[2] = (float)a.pos_cnt;
    }
}

// ------------------------------------------------------------------
// Two different struct types in parallel: each has its own inline,
// and both accumulators must remain live through each other's inline
// merge blocks.  Uses different field types (float vs int) in each struct.

struct SumF { float total; float count_f; };
struct SumI { int count; int abs_count; };

__device__ SumF update_sumf(SumF s, float v) {
    s.total   += v;
    s.count_f += 1.0f;
    return s;
}

__device__ SumI update_sumi(SumI s, float v) {
    s.count     += 1;
    s.abs_count += (v >= 0.0f) ? 1 : 0;  // ternary
    return s;
}

__global__ void parallel_struct_accum(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        SumF sf; sf.total = 0.0f; sf.count_f = 0.0f;
        SumI si; si.count = 0; si.abs_count = 0;
        for (int i = 0; i < n; i++) {
            sf = update_sumf(sf, data[i]);
            si = update_sumi(si, data[i]);
        }
        out[0] = sf.total;
        out[1] = sf.count_f;
        out[2] = (float)si.count;
        out[3] = (float)si.abs_count;
    }
}
