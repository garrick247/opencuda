// Probe: Tricky identifier vs keyword disambiguation
// - Variable named 'n' used in for loop (shadows param? no, same scope)
// - Variable named the same as a struct field 
// - Variable named 'result', 'value', 'index' (common names)
// - re-assignment to loop variable after the loop  
// - Variable 'i' used both as loop var and non-loop var

__global__ void var_shadowing(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int i = tid;  // 'i' as regular var
        int sum = 0;
        for (int j = 0; j < 8; j++) {  // different loop var 'j'
            sum += in[(i + j) % n];
        }
        // i unchanged by loop
        out[i] = sum;
    }
}

__global__ void field_name_clash(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = tid;
        int y = tid * 2;
        // 'x' and 'y' are common struct field names — no clash here
        out[tid] = x + y;
    }
}

// Variable used in both condition and update
__global__ void var_reuse(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = tid;
        int count = 0;
        while (v > 0) {
            v /= 2;
            count++;
        }
        out[tid] = count;
    }
}
