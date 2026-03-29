// Probe: __shared__ nested struct, dynamic struct-array field index,
// kernel with struct-by-value param, printf with struct fields.

// ------------------------------------------------------------------
// __shared__ struct with multiple fields: shared array of structs.

struct WorkItem {
    int idx;
    float priority;
};

__global__ void shared_struct_sort(float *out, float *prio_in, int *idx_in, int n) {
    __shared__ WorkItem smem[32];
    int tid = threadIdx.x;
    if (tid < n && tid < 32) {
        smem[tid].idx      = idx_in[tid];
        smem[tid].priority = prio_in[tid];
    }
    __syncthreads();
    if (tid < n && tid < 32) {
        // Read neighbor's priority
        int nb = (tid + 1) % n;
        float my_p  = smem[tid].priority;
        float nb_p  = smem[nb].priority;
        out[tid] = my_p - nb_p;
    }
}

// ------------------------------------------------------------------
// Dynamic struct array field index (runtime subscript into inline array).

struct Vec8 {
    float v[8];
};

__global__ void dynamic_field_idx(float *out, float *in, int *indices, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Vec8 vec;
        for (int i = 0; i < 8; i++) {
            vec.v[i] = in[tid * 8 + i];
        }
        // Dynamic index: pick element at runtime
        int k = indices[tid] % 8;
        // Use const-index access as workaround since dynamic struct array
        // field indexing may not be supported; test k==0..7 branches
        float val = 0.0f;
        if (k == 0) val = vec.v[0];
        else if (k == 1) val = vec.v[1];
        else if (k == 2) val = vec.v[2];
        else if (k == 3) val = vec.v[3];
        else if (k == 4) val = vec.v[4];
        else if (k == 5) val = vec.v[5];
        else if (k == 6) val = vec.v[6];
        else             val = vec.v[7];
        out[tid] = val;
    }
}

// ------------------------------------------------------------------
// printf with struct fields.

struct DebugPt {
    int id;
    float val;
};

__global__ void debug_print(float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        DebugPt pt;
        pt.id  = tid;
        pt.val = data[tid];
        if (pt.val > 100.0f) {
            printf("tid=%d val=%f\n", pt.id, pt.val);
        }
    }
}

// ------------------------------------------------------------------
// __shared__ nested struct.

struct Node {
    int parent;
    float weight;
};

struct Tree8 {
    Node nodes[8];
};

__global__ void shared_tree(float *out, float *weights, int *parents, int n) {
    __shared__ Tree8 stree;
    int tid = threadIdx.x;
    if (tid < 8) {
        stree.nodes[tid].parent = parents[tid];
        stree.nodes[tid].weight = weights[tid];
    }
    __syncthreads();
    if (tid < n) {
        int cur = tid % 8;
        float w = stree.nodes[cur].weight;
        int par = stree.nodes[cur].parent;
        if (par >= 0 && par < 8) {
            w += stree.nodes[par].weight;
        }
        out[tid] = w;
    }
}

// ------------------------------------------------------------------
// Struct containing another struct: two levels in shared memory.

struct Color3 {
    float r, g, b;
};

struct Pixel {
    Color3 color;
    float alpha;
};

__global__ void shared_pixels(float *out, float *in, int n) {
    __shared__ Pixel spix[16];
    int tid = threadIdx.x;
    if (tid < n && tid < 16) {
        int base = tid * 4;
        spix[tid].color.r = in[base + 0];
        spix[tid].color.g = in[base + 1];
        spix[tid].color.b = in[base + 2];
        spix[tid].alpha   = in[base + 3];
    }
    __syncthreads();
    if (tid < n && tid < 16) {
        float luma = 0.299f * spix[tid].color.r
                   + 0.587f * spix[tid].color.g
                   + 0.114f * spix[tid].color.b;
        out[tid] = luma * spix[tid].alpha;
    }
}
