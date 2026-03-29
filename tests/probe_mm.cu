// Probe: Triple-nested inline struct calls + same param names across fns
// - normalize(cross(a, b)) — result of one inline passed directly to another
// - Function params named same as caller vars (normalize's `v` param = caller `v`)
// - Same function called twice in same loop body with different args
// - 5-field struct

struct V3 { float x, y, z; };
struct Ray { float ox, oy, oz, dx, dy; };  // 5 fields: origin + partial dir

__device__ V3 v3_add(V3 a, V3 b) {
    V3 r; r.x = a.x + b.x; r.y = a.y + b.y; r.z = a.z + b.z; return r;
}

__device__ V3 v3_scale(V3 v, float s) {
    V3 r; r.x = v.x * s; r.y = v.y * s; r.z = v.z * s; return r;
}

__device__ V3 v3_cross(V3 a, V3 b) {
    V3 r;
    r.x = a.y*b.z - a.z*b.y;
    r.y = a.z*b.x - a.x*b.z;
    r.z = a.x*b.y - a.y*b.x;
    return r;
}

__device__ float v3_dot(V3 a, V3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ V3 v3_normalize(V3 v) {
    float len2 = v.x*v.x + v.y*v.y + v.z*v.z;
    if (len2 < 1e-10f) { V3 z; z.x=1.0f; z.y=0.0f; z.z=0.0f; return z; }
    float inv = 1.0f / len2;
    V3 r; r.x = v.x*inv; r.y = v.y*inv; r.z = v.z*inv; return r;
}

// Triple nesting: normalize(cross(a, b))
__global__ void normal_field(float *out, float *as, float *bs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        V3 a; a.x=as[tid*3]; a.y=as[tid*3+1]; a.z=as[tid*3+2];
        V3 b; b.x=bs[tid*3]; b.y=bs[tid*3+1]; b.z=bs[tid*3+2];
        V3 cr = v3_cross(a, b);
        V3 nm = v3_normalize(cr);   // normalize result of cross — triple nesting via temp
        out[tid*3]=nm.x; out[tid*3+1]=nm.y; out[tid*3+2]=nm.z;
    }
}

// Same function called twice with different args in same body
__global__ void gram_schmidt(float *out, float *us, float *vs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        V3 u; u.x=us[tid*3]; u.y=us[tid*3+1]; u.z=us[tid*3+2];
        V3 v; v.x=vs[tid*3]; v.y=vs[tid*3+1]; v.z=vs[tid*3+2];
        float uv = v3_dot(u, v);
        float uu = v3_dot(u, u);  // same fn called twice
        float t = uv / (uu + 1e-10f);
        V3 proj = v3_scale(u, t);
        V3 perp = v3_add(v, v3_scale(proj, -1.0f));  // v - proj
        out[tid*3]=perp.x; out[tid*3+1]=perp.y; out[tid*3+2]=perp.z;
    }
}

// 5-field struct: build Ray from two V3s, accumulate in loop
__global__ void ray_march(float *out, float *origins, float *dirs, float dt, int steps, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Ray ray;
        ray.ox=origins[tid*3]; ray.oy=origins[tid*3+1]; ray.oz=origins[tid*3+2];
        ray.dx=dirs[tid*3];    ray.dy=dirs[tid*3+1];
        V3 pos; pos.x=ray.ox; pos.y=ray.oy; pos.z=ray.oz;
        V3 dir; dir.x=ray.dx; dir.y=ray.dy; dir.z=0.0f;
        for (int i = 0; i < steps; i++) {
            V3 step = v3_scale(dir, dt);
            pos = v3_add(pos, step);
        }
        out[tid*3]=pos.x; out[tid*3+1]=pos.y; out[tid*3+2]=pos.z;
    }
}
