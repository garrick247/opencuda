// Probe: Global device function called with struct argument containing arrays
// - struct with char* member (pointer field)
// - struct passed by reference via pointer, then loaded
// - struct returned from __device__, then field accessed inline

struct Stats {
    float mean;
    float variance;
    float min_val;
    float max_val;
    int count;
};

__device__ Stats compute_stats(float *data, int n) {
    Stats s;
    s.count = n;
    s.mean = 0.0f;
    s.min_val = data[0];
    s.max_val = data[0];
    
    for (int i = 0; i < n; i++) {
        float v = data[i];
        s.mean += v;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
    }
    s.mean /= (float)n;
    
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = data[i] - s.mean;
        var += diff * diff;
    }
    s.variance = var / (float)n;
    return s;
}

__global__ void per_row_stats(float *means, float *variances, float *data,
                               int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        Stats s = compute_stats(data + row * cols, cols);
        means[row] = s.mean;
        variances[row] = s.variance;
    }
}
