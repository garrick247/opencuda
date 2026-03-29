// Probe: struct copy assignment, postfix in subscript, switch-on-cast,
// pointer negative offset, and local array re-init in loop.

// ------------------------------------------------------------------
// Struct copy assignment: a = b where both are structs.
// Tests that all fields are copied correctly.

struct Pair { float x; float y; };

__global__ void struct_copy(float *out, float *xs, float *ys, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        Pair src;
        src.x = xs[tid];
        src.y = ys[tid];
        Pair dst;
        dst = src;              // struct copy assignment
        out[tid * 2 + 0] = dst.x;
        out[tid * 2 + 1] = dst.y;
    }
}

// ------------------------------------------------------------------
// Postfix increment used as array index: data[i++].
// The value of i is used BEFORE the increment.

__global__ void postfix_subscript(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int i = 0;
        int sum = 0;
        while (i < n) {
            sum += data[i++];   // i++ — use then increment
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Switch on a cast expression: switch ((int)(f * 4)).
// Tests that the switch discriminant can be an arbitrary expression.

__global__ void switch_on_cast(int *out, float *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        float f = data[tid];
        int bucket;
        switch ((int)(f * 4.0f)) {
            case 0:  bucket = 0; break;
            case 1:  bucket = 1; break;
            case 2:  bucket = 2; break;
            case 3:  bucket = 3; break;
            default: bucket = 4; break;
        }
        out[tid] = bucket;
    }
}

// ------------------------------------------------------------------
// Pointer with negative offset: ptr[-1], ptr[-2].
// Tests that negative array indices emit correct subtraction.

__global__ void neg_offset(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid >= 2 && tid < n) {
        int v = data[tid] + data[tid - 1] + data[tid - 2];
        out[tid] = v;
    }
}

// ------------------------------------------------------------------
// Local array reinitialized in outer loop.
// Each outer iteration resets the scratch array and reuses it.

__global__ void local_arr_reinit(int *out, int *data, int rows, int cols) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < rows; i++) {
            int scratch[4] = {0, 0, 0, 0};
            for (int j = 0; j < cols && j < 4; j++) {
                scratch[j] = data[i * cols + j];
            }
            // Sum the scratch array
            for (int j = 0; j < 4 && j < cols; j++) {
                total += scratch[j];
            }
        }
        out[0] = total;
    }
}
