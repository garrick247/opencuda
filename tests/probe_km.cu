// Probe: typedef type aliases, enum values in switch/comparison,
// unsigned long long arithmetic,
// address-of-array-element pattern

// typedef int alias
typedef int score_t;
typedef unsigned int uint_t;

__global__ void typedef_ops(score_t *out, score_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        score_t v = in[tid];
        score_t doubled = v * 2;
        out[tid] = doubled;
    }
}

// enum values in switch
typedef enum {
    STATE_IDLE  = 0,
    STATE_RUN   = 1,
    STATE_DONE  = 2,
    STATE_ERROR = 3
} State;

__global__ void state_machine(int *out, int *states, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int s = states[tid];
        int result;
        switch (s) {
            case STATE_IDLE:  result = 0;    break;
            case STATE_RUN:   result = 100;  break;
            case STATE_DONE:  result = 200;  break;
            default:          result = -1;   break;
        }
        out[tid] = result;
    }
}

// unsigned long long arithmetic
__global__ void ull_arithmetic(unsigned long long *out,
                               unsigned long long *a,
                               unsigned long long *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned long long va = a[tid];
        unsigned long long vb = b[tid];
        out[tid * 3]     = va + vb;
        out[tid * 3 + 1] = va * vb;
        out[tid * 3 + 2] = (va > vb) ? va - vb : vb - va;  // abs diff
    }
}

// Address-of array element: write to a specific position via pointer
__global__ void addr_of_elem(int *arr, int n, int target_idx, int target_val) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int *p = &arr[target_idx];
        *p = target_val;
    }
}

// uint_t typedef used for unsigned arithmetic
__global__ void uint_typedef(uint_t *out, uint_t *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        uint_t v = in[tid];
        out[tid] = ~v & 0xFFFF;   // flip low 16 bits
    }
}
