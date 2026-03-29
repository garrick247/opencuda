// Probe: complex inline device function patterns — multiple returns
// that exercise the single-return-point limitation, recursive-like
// structures, and struct field mutation chains.

struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

struct Ray {
    float ox, oy, oz;   // origin
    float dx, dy, dz;   // direction
};

// ------------------------------------------------------------------
// AABB ray intersection test (classic slab method).

__device__ float aabb_intersect(struct AABB b, struct Ray r) {
    float tx1 = (b.min_x - r.ox) / r.dx;
    float tx2 = (b.max_x - r.ox) / r.dx;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);

    float ty1 = (b.min_y - r.oy) / r.dy;
    float ty2 = (b.max_y - r.oy) / r.dy;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));

    float tz1 = (b.min_z - r.oz) / r.dz;
    float tz2 = (b.max_z - r.oz) / r.dz;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));

    return (tmax >= tmin) ? tmin : -1.0f;
}

__global__ void ray_aabb(float *out, struct AABB *boxes, struct Ray *rays, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = aabb_intersect(boxes[tid], rays[tid]);
    }
}

// ------------------------------------------------------------------
// Struct returned by value (multi-field).

struct Stats {
    float mean;
    float var;
    int count;
};

__device__ struct Stats compute_stats(float *data, int start, int count) {
    struct Stats s;
    s.count = count;
    float sum = 0.0f, sum2 = 0.0f;
    for (int i = start; i < start + count; i++) {
        float v = data[i];
        sum += v;
        sum2 += v * v;
    }
    s.mean = sum / (float)count;
    s.var  = sum2 / (float)count - s.mean * s.mean;
    return s;
}

__global__ void stats_kernel(float *mean_out, float *var_out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid * 4 < n) {
        struct Stats s = compute_stats(data, tid * 4, 4);
        mean_out[tid] = s.mean;
        var_out[tid]  = s.var;
    }
}

// ------------------------------------------------------------------
// Newton-Raphson iteration for sqrt approximation.

__device__ float my_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    float r = x * 0.5f;  // initial guess
    // 3 Newton-Raphson iterations: r = (r + x/r) / 2
    r = (r + x / r) * 0.5f;
    r = (r + x / r) * 0.5f;
    r = (r + x / r) * 0.5f;
    return r;
}

__global__ void nr_sqrt(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) out[tid] = my_sqrt(in[tid]);
}

// ------------------------------------------------------------------
// Polynomial evaluation (Horner's method).

__device__ float horner4(float x, float a, float b, float c, float d) {
    // a*x^3 + b*x^2 + c*x + d = ((a*x + b)*x + c)*x + d
    return ((a * x + b) * x + c) * x + d;
}

__global__ void poly_eval(float *out, float *x, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = horner4(x[tid], 1.0f, -3.0f, 3.0f, -1.0f);  // (x-1)^3
    }
}

// ------------------------------------------------------------------
// Binary search in sorted array.

__device__ int binary_search(int *arr, int n, int target) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

__global__ void bsearch_kernel(int *out, int *arr, int *queries, int n, int m) {
    int tid = threadIdx.x;
    if (tid < m) {
        out[tid] = binary_search(arr, n, queries[tid]);
    }
}

// ------------------------------------------------------------------
// Struct array of function arguments (composite computation).

__global__ void composite_kernel(float *out, float *a, float *b,
                                  float *c, float *d, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float av = a[tid], bv = b[tid], cv = c[tid], dv = d[tid];
        // Compute multiple derived values
        float p = av * bv;
        float q = cv + dv;
        float r = p / (q + 1.0f);
        float s = horner4(r, av, bv, cv, dv);
        out[tid] = s + my_sqrt(fabsf(p - q));
    }
}
