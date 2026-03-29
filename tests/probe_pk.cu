// Probe: for-loop comma update, const/restrict qualifiers, multiple
// assignments in nested &&, and struct-returning fn called with array element.

// ------------------------------------------------------------------
// for-loop with comma operator in update: i++, j--.
// Tests that the parser handles multiple expressions in the for-update.

__global__ void two_pointer_sum(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0, j = n - 1; i < j; i++, j--) {
            sum += data[i] + data[j];
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// const and __restrict__ on pointer parameters.
// These qualifiers should be silently accepted.

__global__ void const_restrict(int *__restrict__ out,
                                const int *__restrict__ data,
                                int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = data[tid] * 2;
    }
}

// ------------------------------------------------------------------
// Multiple &&-assignments in nested &&:
// `while (i < n && (a = arr1[i]) > 0 && (b = arr2[i]) > 0)`.
// a is rebound in the first land_rhs, b is rebound in the second land_rhs.
// Both must be multi-def after the fix.

__global__ void nested_and_assign(int *out, int *arr1, int *arr2, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0, a = 0, b = 0;
        int i = 0;
        while (i < n && (a = arr1[i]) > 0 && (b = arr2[i]) > 0) {
            sum += a + b;
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Ternary assignment in && RHS:
// `while (i < n && (v = x > 0 ? x : -x) < threshold)`.

__global__ void ternary_assign_in_and(int *out, int *data, int n, int threshold) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0, v = 0;
        int i = 0;
        while (i < n && (v = (data[i] > 0 ? data[i] : -data[i])) < threshold) {
            sum += v;
            i++;
        }
        out[0] = sum;
    }
}

// ------------------------------------------------------------------
// Struct-returning device function with struct array element input.
// Tests that struct return from a function called with a struct array element
// works correctly.

struct MinMax {
    int lo, hi;
};

__device__ MinMax clamp_range(int v, int lo, int hi) {
    MinMax r;
    r.lo = (v < lo) ? lo : v;
    r.hi = (v > hi) ? hi : v;
    return r;
}

__global__ void struct_return_array(int *out, int *data, int n, int lo, int hi) {
    int tid = threadIdx.x;
    if (tid < n) {
        MinMax r = clamp_range(data[tid], lo, hi);
        out[tid] = r.lo + r.hi;
    }
}

// ------------------------------------------------------------------
// Assignment to loop variable in for-update via compound assignment.
// `for (int i = n-1; i >= 0; i -= 2)` — step by 2.

__global__ void step_by_two(int *out, int *data, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = n - 1; i >= 0; i -= 2) {
            sum += data[i];
        }
        out[0] = sum;
    }
}
