// Probe: float variables modified before for-loop continue,
// double-precision accumulator with continue,
// mixed int+float state surviving continue,
// continue after compound float assignment

// Float accumulator that skips negatives
__global__ void float_sum_skip(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float pos_sum = 0.0f;
        float neg_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v < 0.0f) {
                neg_sum += v;   // accumulate before continue
                continue;
            }
            pos_sum += v;
        }
        out[0] = pos_sum;
        out[1] = neg_sum;
    }
}

// Running max with continue
__global__ void running_max_continue(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float cur_max = in[0];
        int max_idx = 0;
        for (int i = 1; i < n; i++) {
            float v = in[i];
            if (v <= cur_max) continue;  // no mutation before continue
            cur_max = v;
            max_idx = i;
        }
        out[0] = cur_max;
        out[1] = (float)max_idx;
    }
}

// Mixed int+float state: both updated before continue
__global__ void mixed_state_continue(float *fout, int *iout, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float fsum = 0.0f;
        int icount = 0;
        float fskip = 0.0f;
        int iskip = 0;
        for (int i = 0; i < n; i++) {
            float v = fin[i];
            if (v < 0.0f) {
                fskip += v;
                iskip++;
                continue;
            }
            fsum += v;
            icount++;
        }
        fout[0] = fsum;
        fout[1] = fskip;
        iout[0] = icount;
        iout[1] = iskip;
    }
}

// Compound float update (*= before continue)
__global__ void float_scale_continue(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float product = 1.0f;
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float v = in[i];
            if (v == 0.0f) {
                sum += 1.0f;   // track zeros separately
                continue;
            }
            product *= v;
            sum += v;
        }
        out[0] = product;
        out[1] = sum;
    }
}
