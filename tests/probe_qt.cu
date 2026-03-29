// Probe: __device__ functions accessing global vars directly, do-while
// with struct fields, ternary with store side effects, and global
// struct used by multiple kernels with different access patterns.

// ------------------------------------------------------------------
// __device__ function accessing a global var directly.

__device__ int g_call_count;

__device__ void increment_call_count() {
    atomicAdd(&g_call_count, 1);
}

__global__ void call_counting(int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        if (data[tid] > 0) {
            increment_call_count();
        }
    }
}

__global__ void read_call_count(int *out) {
    if (threadIdx.x == 0) {
        out[0] = g_call_count;
    }
}

// ------------------------------------------------------------------
// do-while with struct fields as loop condition variable.

struct Iter {
    float val;
    int   step;
};

__global__ void do_while_struct(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Iter it;
        it.val  = in[tid];
        it.step = 0;
        do {
            it.val  *= 0.5f;
            it.step++;
        } while (it.val > 0.01f && it.step < 20);
        out[tid * 2 + 0] = it.val;
        out[tid * 2 + 1] = (float)it.step;
    }
}

// ------------------------------------------------------------------
// Ternary with different global struct field stores on each arm.

struct TwoCounters {
    int pos;
    int neg;
};

__device__ TwoCounters g_tc;

__global__ void ternary_store(int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            int v = data[i];
            // Ternary as expression on LHS is not valid, so use if/else.
            // But ternary on the value being stored:
            int delta = (v > 0) ? 1 : 0;
            g_tc.pos += delta;
            g_tc.neg += (1 - delta);
        }
    }
}

// ------------------------------------------------------------------
// Global struct array: read-then-write same element in same kernel.

struct Cell {
    float data;
    int   flag;
};

__device__ Cell g_cells[8];

__global__ void update_cells(float threshold) {
    int tid = threadIdx.x;
    if (tid < 8) {
        float v = g_cells[tid].data;
        if (v > threshold) {
            g_cells[tid].data = v * 0.9f;
            g_cells[tid].flag = 1;
        } else {
            g_cells[tid].flag = 0;
        }
    }
}

// ------------------------------------------------------------------
// Complex expression: struct field used in multiple sub-expressions.

struct Rect {
    float x, y, w, h;
};

__device__ float rect_area(Rect r) {
    return r.w * r.h;
}

__device__ float rect_perimeter(Rect r) {
    return 2.0f * (r.w + r.h);
}

__global__ void rect_properties(float *out_area, float *out_peri,
                                 float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Rect r;
        r.x = data[tid * 4 + 0];
        r.y = data[tid * 4 + 1];
        r.w = data[tid * 4 + 2];
        r.h = data[tid * 4 + 3];
        out_area[tid] = rect_area(r);
        out_peri[tid] = rect_perimeter(r);
    }
}

// ------------------------------------------------------------------
// Loop with struct field as both loop variable and accumulator.

__global__ void sliding_window(float *out, float *in, int win, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        Iter w;
        w.val  = 0.0f;
        w.step = 0;
        for (int i = 0; i < n; i++) {
            w.val  += in[i];
            w.step++;
            if (w.step > win) {
                w.val  -= in[i - win];
                w.step--;
            }
            out[i] = w.val / (float)w.step;
        }
    }
}
