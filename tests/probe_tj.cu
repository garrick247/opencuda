// Probe: preprocessor patterns — #ifdef/#ifndef, multi-token macros,
// do-while(0) macro idiom, and stringification-adjacent patterns.

#define THREADS_PER_BLOCK 256
#define MAX_ITER 32
#define EPSILON 1e-6f
#define ALPHA 0.1f
#define BETA  0.9f

// Macro that expands to an expression.
#define SQ(x)       ((x)*(x))
#define CUBE(x)     ((x)*(x)*(x))
#define ABS(x)      ((x) < 0 ? -(x) : (x))
#define MAX(a,b)    ((a) > (b) ? (a) : (b))
#define MIN(a,b)    ((a) < (b) ? (a) : (b))
#define CLAMP(v,lo,hi) MIN(MAX(v, lo), hi)
#define LERP(a,b,t) ((a) + ((b)-(a))*(t))

// Macro that acts like a statement (do-while(0) idiom).
#define SWAP(a, b, tmp)  do { (tmp) = (a); (a) = (b); (b) = (tmp); } while(0)
#define INC_CLAMP(v, hi) do { if ((v) < (hi)) (v)++; } while(0)

// ------------------------------------------------------------------
// Use constant macros.

__global__ void macro_consts(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = ALPHA * v + BETA;
        if (ABS(r) < EPSILON) r = 0.0f;
        out[tid] = CLAMP(r, -1.0f, 1.0f);
    }
}

// ------------------------------------------------------------------
// Use expression macros in loops.

__global__ void macro_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float acc = 0.0f;
        for (int i = 0; i < MAX_ITER; i++) {
            float x = v + (float)i * EPSILON;
            acc += SQ(x) + CUBE(x) * ALPHA;
        }
        out[tid] = acc;
    }
}

// ------------------------------------------------------------------
// Nested macro calls.

__global__ void macro_nested(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float v = in[tid];
        float r = LERP(MIN(v, 1.0f), MAX(v, -1.0f), ALPHA);
        out[tid] = r;
    }
}

// ------------------------------------------------------------------
// SWAP macro using do-while(0).

__global__ void macro_swap(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int a = in[tid * 2 + 0];
        int b = in[tid * 2 + 1];
        int tmp;
        if (a > b) {
            SWAP(a, b, tmp);
        }
        out[tid * 2 + 0] = a;
        out[tid * 2 + 1] = b;
    }
}

// ------------------------------------------------------------------
// INC_CLAMP macro in loop.

__global__ void macro_inc_clamp(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        for (int i = 0; i < 10; i++) {
            if (v + i > 5) {
                INC_CLAMP(count, 5);
            }
        }
        out[tid] = count;
    }
}

// ------------------------------------------------------------------
// Macro that expands to a multi-op expression.

#define DOT3(ax,ay,az, bx,by,bz) ((ax)*(bx) + (ay)*(by) + (az)*(bz))

__global__ void macro_dot3(float *out, float *a, float *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float ax = a[tid*3+0], ay = a[tid*3+1], az = a[tid*3+2];
        float bx = b[tid*3+0], by = b[tid*3+1], bz = b[tid*3+2];
        out[tid] = DOT3(ax, ay, az, bx, by, bz);
    }
}

// ------------------------------------------------------------------
// Macros for loop bounds.

#define FOR_EACH(i, n) for (int i = 0; i < (n); i++)
#define FOR_TILE(i, start, end) for (int i = (start); i < (end); i++)

__global__ void macro_for_each(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int acc = 0;
        FOR_EACH(i, 8) {
            acc += in[tid] + i;
        }
        FOR_TILE(j, 2, 6) {
            acc -= j;
        }
        out[tid] = acc;
    }
}
