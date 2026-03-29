// Probe: combinations of features — struct with pointer fields,
// local struct arrays, loop modifying struct fields, and
// warp-level operations with early return.

struct Stats { float sum, sum2, min_v, max_v; int count; };

// ------------------------------------------------------------------
// Compute running statistics in a struct.

__global__ void running_stats(struct Stats *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Stats s;
        s.sum   = 0.0f;
        s.sum2  = 0.0f;
        s.min_v =  1e30f;
        s.max_v = -1e30f;
        s.count = 0;

        for (int i = 0; i < k; i++) {
            float v = in[tid * k + i];
            s.sum   += v;
            s.sum2  += v * v;
            s.min_v  = (v < s.min_v) ? v : s.min_v;
            s.max_v  = (v > s.max_v) ? v : s.max_v;
            s.count++;
        }

        out[tid].sum   = s.sum;
        out[tid].sum2  = s.sum2;
        out[tid].min_v = s.min_v;
        out[tid].max_v = s.max_v;
        out[tid].count = s.count;
    }
}

// ------------------------------------------------------------------
// Local array of small structs.

struct Point2D { float x, y; };

__global__ void local_struct_array(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Point2D pts[4];
        float v = in[tid];
        // Fill array
        for (int i = 0; i < 4; i++) {
            pts[i].x = v + (float)i;
            pts[i].y = v - (float)i;
        }
        // Sum x and y
        float sx = 0.0f, sy = 0.0f;
        for (int i = 0; i < 4; i++) {
            sx += pts[i].x;
            sy += pts[i].y;
        }
        out[tid * 2 + 0] = sx;
        out[tid * 2 + 1] = sy;
    }
}

// ------------------------------------------------------------------
// Struct modification in loop with early exit.

__global__ void struct_in_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Stats s;
        s.sum   = 0.0f;
        s.count = 0;
        s.min_v = in[tid];
        s.max_v = in[tid];

        for (int i = 0; i < 16; i++) {
            float v = in[tid] * (float)(i + 1);
            if (v > 1000.0f) break;
            s.sum   += v;
            s.count += 1;
            if (v < s.min_v) s.min_v = v;
            if (v > s.max_v) s.max_v = v;
        }

        out[tid * 3 + 0] = s.sum;
        out[tid * 3 + 1] = s.min_v;
        out[tid * 3 + 2] = s.max_v;
    }
}

// ------------------------------------------------------------------
// Warp reduce with struct accumulation.

struct WarpSum { int sum, count; };

__device__ struct WarpSum warp_reduce_sum(struct WarpSum ws) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        ws.sum   += __shfl_down_sync(0xFFFFFFFF, ws.sum,   offset);
        ws.count += __shfl_down_sync(0xFFFFFFFF, ws.count, offset);
    }
    return ws;
}

__global__ void warp_struct_reduce(int *out, int *in, int n) {
    int tid = threadIdx.x;
    struct WarpSum ws;
    ws.sum   = (tid < n) ? in[tid] : 0;
    ws.count = (tid < n) ? 1 : 0;

    ws = warp_reduce_sum(ws);

    if ((tid & 31) == 0 && tid < n) {
        out[tid >> 5] = (ws.count > 0) ? ws.sum / ws.count : 0;
    }
}

// ------------------------------------------------------------------
// Device function called multiple times — accumulate into struct.

__device__ void update_stats(struct Stats *s, float v) {
    s->sum   += v;
    s->sum2  += v * v;
    s->count += 1;
    if (v < s->min_v) s->min_v = v;
    if (v > s->max_v) s->max_v = v;
}

__global__ void accum_stats_kernel(struct Stats *out, float *in, int n, int k) {
    int tid = threadIdx.x;
    if (tid < n) {
        struct Stats s;
        s.sum = s.sum2 = 0.0f;
        s.min_v =  1e30f;
        s.max_v = -1e30f;
        s.count = 0;
        for (int i = 0; i < k; i++) {
            update_stats(&s, in[tid * k + i]);
        }
        out[tid] = s;
    }
}
