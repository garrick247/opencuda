// Probe: complex composite — nested struct with 3D fields, global array,
// shared memory, and __device__ functions doing reads/writes through ptrs.
// This is the "torture test" combining all previously-fixed bug classes.

struct Vec3 {
    float x, y, z;
};

struct Transform {
    Vec3  pos;
    Vec3  rot;
    float scale;
};

__device__ Transform g_xforms[8];

// ------------------------------------------------------------------
// Read nested struct fields through pointer-to-Transform.

__device__ float xform_mag(Transform *t) {
    float px = t->pos.x, py = t->pos.y, pz = t->pos.z;
    float rx = t->rot.x, ry = t->rot.y, rz = t->rot.z;
    return (px*px + py*py + pz*pz) + (rx*rx + ry*ry + rz*rz);
}

// ------------------------------------------------------------------
// Write nested struct fields through pointer.

__device__ void xform_scale(Transform *t, float s) {
    t->pos.x *= s;
    t->pos.y *= s;
    t->pos.z *= s;
    t->scale *= s;
}

// ------------------------------------------------------------------
// Kernel: read from global g_xforms[], apply scale, write back.

__global__ void apply_scale(float *scales, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        xform_scale(&g_xforms[tid], scales[tid]);
    }
}

// ------------------------------------------------------------------
// Kernel: compute magnitude of each global transform.

__global__ void compute_mags(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        out[tid] = xform_mag(&g_xforms[tid]);
    }
}

// ------------------------------------------------------------------
// Kernel: copy from global to shared, transform in shared, write back.

__global__ void shared_xform_pass(float *scales, int n) {
    __shared__ Transform sxforms[8];
    int tid = threadIdx.x;

    // Load global → shared (field by field)
    if (tid < 8) {
        sxforms[tid].pos.x  = g_xforms[tid].pos.x;
        sxforms[tid].pos.y  = g_xforms[tid].pos.y;
        sxforms[tid].pos.z  = g_xforms[tid].pos.z;
        sxforms[tid].rot.x  = g_xforms[tid].rot.x;
        sxforms[tid].rot.y  = g_xforms[tid].rot.y;
        sxforms[tid].rot.z  = g_xforms[tid].rot.z;
        sxforms[tid].scale  = g_xforms[tid].scale;
    }
    __syncthreads();

    // Transform in shared
    if (tid < n && tid < 8) {
        float s = scales[tid];
        sxforms[tid].pos.x *= s;
        sxforms[tid].pos.y *= s;
        sxforms[tid].pos.z *= s;
        sxforms[tid].scale *= s;
    }
    __syncthreads();

    // Write shared → global
    if (tid < 8) {
        g_xforms[tid].pos.x = sxforms[tid].pos.x;
        g_xforms[tid].pos.y = sxforms[tid].pos.y;
        g_xforms[tid].pos.z = sxforms[tid].pos.z;
        g_xforms[tid].scale = sxforms[tid].scale;
    }
}

// ------------------------------------------------------------------
// Fill g_xforms from flat array (tests write path for all fields).

__global__ void fill_xforms(float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        int b = tid * 7;
        g_xforms[tid].pos.x = data[b + 0];
        g_xforms[tid].pos.y = data[b + 1];
        g_xforms[tid].pos.z = data[b + 2];
        g_xforms[tid].rot.x = data[b + 3];
        g_xforms[tid].rot.y = data[b + 4];
        g_xforms[tid].rot.z = data[b + 5];
        g_xforms[tid].scale = data[b + 6];
    }
}

// ------------------------------------------------------------------
// Read all fields of g_xforms into flat array (tests read path).

__global__ void read_xforms(float *out, int n) {
    int tid = threadIdx.x;
    if (tid < n && tid < 8) {
        int b = tid * 7;
        out[b + 0] = g_xforms[tid].pos.x;
        out[b + 1] = g_xforms[tid].pos.y;
        out[b + 2] = g_xforms[tid].pos.z;
        out[b + 3] = g_xforms[tid].rot.x;
        out[b + 4] = g_xforms[tid].rot.y;
        out[b + 5] = g_xforms[tid].rot.z;
        out[b + 6] = g_xforms[tid].scale;
    }
}
