// Probe: complex address-of patterns — &struct_field, &array[i],
// pointer to pointer deref pattern (**pp = val)

struct Node {
    int val;
    int next;  // index into array, not actual pointer
};

__global__ void linked_list_sum(int *out, Node *nodes, int start, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int idx = start;
        int count = 0;
        while (idx >= 0 && idx < n && count < n) {
            sum += nodes[idx].val;
            idx = nodes[idx].next;
            count++;
        }
        out[0] = sum;
    }
}

// Pointer-to-struct field (address-of member)
__device__ void increment_field(int *field_ptr) {
    (*field_ptr)++;
}

__global__ void field_ptr(int *out, Node *nodes, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Node *p = &nodes[tid];
        increment_field(&p->val);
        increment_field(&p->next);
        out[tid] = p->val + p->next;
    }
}

// Double pointer: passing pointer to pointer
__device__ void update_ptr(float **pp, float *new_ptr) {
    *pp = new_ptr;
}

__global__ void double_ptr(float *out, float *a, float *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float *p = a;
        if (sel[tid]) {
            update_ptr(&p, b);
        }
        out[tid] = p[tid];
    }
}
