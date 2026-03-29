// Probe: Complex struct patterns
// - Struct with function pointer member (parsing only)
// - Struct inheritance via embedding (C-style)
// - Struct with bitfield members (should parse and ignore bitfield size)
// - Array of structs with specific indexing patterns

struct Node {
    float val;
    int left;   // index of left child (-1 if leaf)
    int right;  // index of right child (-1 if leaf)
};

__device__ float tree_eval(Node *nodes, float x) {
    int cur = 0;
    while (cur >= 0) {
        float v = nodes[cur].val;
        if (x < v) {
            int next = nodes[cur].left;
            if (next < 0) return v;
            cur = next;
        } else {
            int next = nodes[cur].right;
            if (next < 0) return v;
            cur = next;
        }
    }
    return 0.0f;
}

__global__ void eval_tree(float *out, float *in, Node *tree, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = tree_eval(tree, in[tid]);
    }
}
