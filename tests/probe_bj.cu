// Probe: Patterns that expose codegen issues rather than parser issues
// - Very long variable names
// - Shadows: local var with same name as param
// - Two kernels that both modify a global struct  
// - Multiple return paths in device function used in condition

__device__ int sign_of(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

// Use device function result directly in arithmetic expression (no temp var)
__global__ void direct_func_arith(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // sign_of() result used directly without assigning to variable
        out[tid] = sign_of(in[tid]) * 100 + sign_of(-in[tid]) * 200;
    }
}

// Long variable names (stress register naming)
__global__ void long_var_names(float *output_buffer, float *input_buffer, int total_element_count) {
    int thread_global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_global_index < total_element_count) {
        float input_element_value = input_buffer[thread_global_index];
        float processed_element_value = input_element_value * input_element_value + 1.0f;
        output_buffer[thread_global_index] = processed_element_value;
    }
}

// Local var shadows param name
__global__ void shadow_param(int *out, int n) {
    int n_local = n * 2;  // 'n_local' doesn't shadow, but using same name pattern
    int tid = threadIdx.x;
    int n2 = n_local;     // just makes sure 'n' is still accessible
    if (tid < n) {
        out[tid] = n2 - n;  // should be n
    }
}
