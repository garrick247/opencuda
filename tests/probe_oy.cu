// Probe: multi-level pointer deref, array of structs write, prefix ++/-- as
// statement (not expression), and compound assign with type promotion.

// ------------------------------------------------------------------
// Multi-level pointer dereference: **pp = val.
// Tests two levels of indirection in a load chain.

__global__ void double_deref(int *out, int **pp, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = *(*pp + tid);   // deref pp, offset, deref element
    }
}

// ------------------------------------------------------------------
// Array of structs write: set all fields of each struct element.

struct Point3 { float x; float y; float z; };

__global__ void aos_write3(Point3 *pts, float *xs, float *ys, float *zs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        pts[tid].x = xs[tid];
        pts[tid].y = ys[tid];
        pts[tid].z = zs[tid];
    }
}

// ------------------------------------------------------------------
// Prefix increment/decrement used as a statement (not as expression value).

__global__ void prefix_stmt(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int count = 0;
        int neg = 0;
        for (int i = 0; i < n; i++) {
            if (data[i] > 0) ++count;
            else             --neg;
        }
        out[0] = count;
        out[1] = neg;   // neg is negative: number of non-positive values negated
    }
}

// ------------------------------------------------------------------
// Compound assignment with type promotion: float += int, int += short cast.

__global__ void compound_promo(float *fout, int *iout, float *fdata, int *idata, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float fsum = 0.0f;
        int   isum = 0;
        for (int i = 0; i < n; i++) {
            fsum += (int)fdata[i];   // float += int (promotes int to float)
            isum += (int)fdata[i];   // int   += int (no promotion needed)
        }
        fout[0] = fsum;
        iout[0] = isum;
    }
}

// ------------------------------------------------------------------
// Warp-level min reduction using shfl_down_sync.
// Classic warp reduction pattern — tests warp intrinsic in reduction loop.

__global__ void warp_min_reduce(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = data[tid];
        // Warp reduction: reduce across 32 lanes
        for (int delta = 16; delta >= 1; delta >>= 1) {
            int other = __shfl_down_sync(0xffffffff, v, delta);
            if (v > other) v = other;
        }
        if ((tid & 31) == 0) {
            out[tid >> 5] = v;
        }
    }
}
