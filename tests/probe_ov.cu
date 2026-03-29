// Probe: switch fall-through, struct return from device fn,
// nested continue, and do-while continue.

// ------------------------------------------------------------------
// Switch fall-through: cases 1 and 2 share the same body.
// No break between case 1 and case 2 — fall-through must work.

__global__ void switch_fallthrough(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        int result = 0;
        switch (v) {
            case 0:
                result = 10;
                break;
            case 1:
            case 2:
                result = 20;
                break;
            case 3:
                result = 30;
                break;
            default:
                result = -1;
                break;
        }
        out[tid] = result;
    }
}

// ------------------------------------------------------------------
// Struct returned by value from __device__ function.
// Tests that struct return is decomposed into scalar registers.

struct Vec2 { float x; float y; };

__device__ Vec2 make_vec2(float x, float y) {
    Vec2 v;
    v.x = x;
    v.y = y;
    return v;
}

__global__ void struct_return(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec2 v = make_vec2(xs[tid], ys[tid]);
        out[tid * 2 + 0] = v.x;
        out[tid * 2 + 1] = v.y;
    }
}

// ------------------------------------------------------------------
// Conditional continue in for loop: skip negative values.

__global__ void skip_negatives(float *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (data[i] < 0.0f) continue;
            sum += data[i];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Nested loops: continue on inner loop, break on outer.
// Outer loop breaks when a sentinel (-1) is found.

__global__ void nested_continue(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int v = data[i * cols + j];
                if (v == 0) continue;   // skip zeros in inner loop
                sum += v;
            }
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// do-while with continue: runs body at least once, continue skips to test.

__global__ void do_while_continue(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        do {
            if (data[i] < 0) {
                i++;
                continue;
            }
            sum += data[i];
            i++;
        } while (i < n);
        out[0] = sum;
    }
}
