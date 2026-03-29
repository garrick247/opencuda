// Probe: More C++ syntax patterns in CUDA
// - operator keyword (should fail gracefully or be skipped)
// - throw/try/catch (should be skipped)
// - class (should be treated like struct)
// - new/delete (should fail gracefully)
// - Reference type param T& (parse as T, ignore &)

// class treated as struct
class Vector2 {
public:
    float x, y;
};

__device__ float vec2_length(Vector2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__global__ void vec_lengths(float *out, Vector2 *vecs, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = vec2_length(vecs[tid]);
    }
}
