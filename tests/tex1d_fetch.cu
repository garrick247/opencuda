// Test: tex1Dfetch, tex2D, tex3D intrinsics
__global__ void kernel_tex1d(float *out, long long texObj, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = tex1Dfetch(texObj, idx);
        out[idx] = val;
    }
}

__global__ void kernel_tex2d(float *out, long long texObj, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < w && y < h) {
        float fx = (float)x;
        float fy = (float)y;
        float val = tex2D(texObj, fx, fy);
        out[y * w + x] = val;
    }
}

__global__ void kernel_tex3d(float *out, long long texObj, int w, int h, int d) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    if (x < w && y < h && z < d) {
        float fx = (float)x;
        float fy = (float)y;
        float fz = (float)z;
        float val = tex3D(texObj, fx, fy, fz);
        out[z * w * h + y * w + x] = val;
    }
}
