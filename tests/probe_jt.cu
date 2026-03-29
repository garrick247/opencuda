// Probe: inline device function whose local variable name matches caller variable
// After the call, caller's variable must NOT be reset to pre-call value

__device__ int compute(int val) {
    int result = val * 2;   // 'result' is local to compute
    return result;
}

// If result was modified before calling compute(), it must survive the call
__global__ void caller_var_survives(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int result = 0;       // caller's 'result'
        for (int i = 0; i < n; i++) {
            result += in[i];  // modify result before call
            int doubled = compute(in[i]);  // inlined: declares 'result' locally
            result += doubled;             // result must still be the accumulated sum
        }
        *out = result;
    }
}

// Device function with same param name as caller variable
__device__ int scale_add(int val, int scale) {
    int val2 = val * scale;  // 'val2' is local
    return val2 + val;       // returned
}

__global__ void param_name_collision(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];    // caller's 'val'
        int val2 = val + 1;   // caller's 'val2'
        // scale_add has params 'val' and 'scale', and local 'val2'
        int r = scale_add(val, 3);  // inlined: binds val=val, scale=3, declares val2 internally
        // After call: caller's val and val2 must be unchanged
        out[tid] = r + val + val2;  // should be (val*3+val) + val + (val+1) = val*5 + val + 1
    }
}

// Device function declared before kernel — forward-decl already tested,
// but inlining with accumulator variable named same as loop body local
__device__ int abs_val(int x) {
    int result = x < 0 ? -x : x;   // local 'result'
    return result;
}

__global__ void accum_same_name(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int result = 0;   // outer accumulator named 'result'
        for (int i = 0; i < n; i++) {
            result += abs_val(in[i]);  // abs_val declares 'result' internally
        }
        *out = result;   // should be sum of abs values
    }
}
