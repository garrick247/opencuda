// Probe: deeply nested control flow, goto/label patterns,
// multiple nested loops with break/continue, complex phi placement,
// and struct-field conditional updates.

// ------------------------------------------------------------------
// Deeply nested if-else: 4 levels deep.

__global__ void deep_nested_if(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int r;
        if (v > 100) {
            if (v > 200) {
                if (v > 300) {
                    if (v > 400) r = 4;
                    else         r = 3;
                } else          r = 2;
            } else              r = 1;
        } else if (v < -100) {
            if (v < -200) r = -2;
            else          r = -1;
        } else {
            r = 0;
        }
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// Multiple break + continue in nested loops.

__global__ void nested_break_continue(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int found_row = -1, found_col = -1;
        for (int r = 0; r < rows; r++) {
            int found_in_row = 0;
            for (int c = 0; c < cols; c++) {
                if (data[r * cols + c] < 0) continue;
                if (data[r * cols + c] > 1000) {
                    found_row = r;
                    found_col = c;
                    found_in_row = 1;
                    break;
                }
            }
            if (found_in_row) break;
        }
        out[0] = found_row;
        out[1] = found_col;
    }
}

// ------------------------------------------------------------------
// Struct with conditional field updates at multiple if-merge points.

struct State {
    int   phase;
    float energy;
    float momentum;
};

__global__ void state_machine(State *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        State s;
        s.phase    = 0;
        s.energy   = in[tid];
        s.momentum = 0.0f;

        for (int step = 0; step < 10; step++) {
            if (s.phase == 0) {
                s.energy += 1.0f;
                if (s.energy > 5.0f) {
                    s.phase = 1;
                    s.momentum = s.energy * 0.1f;
                }
            } else if (s.phase == 1) {
                s.energy   -= 0.5f;
                s.momentum *= 1.1f;
                if (s.energy < 1.0f) {
                    s.phase = 2;
                }
            } else {
                s.energy   *= 0.9f;
                s.momentum  = 0.0f;
            }
        }

        out[tid] = s;
    }
}

// ------------------------------------------------------------------
// Goto + label: jump over a block.

__global__ void goto_skip(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int result = 0;
        if (v < 0) goto neg_path;
        result = v * 2;
        goto done;
    neg_path:
        result = -v;
    done:
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Complex phi: variable assigned in 3 different branches, used after.

__global__ void three_way_assign(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = data[tid];
        float r;
        if (v > 1.0f) {
            r = v * v;
        } else if (v > 0.0f) {
            r = v;
        } else {
            r = 0.0f;
        }
        out[tid] = r * 2.0f;  // uses r from all 3 paths
    }
}

// ------------------------------------------------------------------
// Loop with struct field driving the condition.

__global__ void field_driven_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        State s;
        s.phase    = 0;
        s.energy   = in[tid];
        s.momentum = 1.0f;
        int iter = 0;
        while (s.energy > 0.1f && iter < 100) {
            s.energy   -= s.momentum * 0.05f;
            s.momentum *= 0.99f;
            iter++;
        }
        out[tid] = s.energy;
    }
}
