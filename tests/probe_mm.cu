// Probe: Switch fallthrough, nested switch, large case counts
// - Switch with explicit fallthrough (no break between cases)
// - Switch inside while loop (break exits switch not loop)
// - Nested switch
// - Switch on expression with many cases
// - Switch followed by more code using the set variable

__global__ void switch_fallthrough(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid] % 4;
        int val = 0;
        switch (x) {
            case 0:
            case 1:
                val = 10;  // cases 0 and 1 both get 10 (fallthrough)
                break;
            case 2:
                val = 20;
                break;
            default:
                val = 30;
        }
        out[tid] = val;
    }
}

// Switch inside while loop: break exits the switch, not the while
__global__ void switch_in_while(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int i = 0;
        while (i < n) {
            switch (i % 3) {
                case 0: sum += i; break;
                case 1: sum -= i; break;
                default: break;
            }
            i++;
        }
        out[0] = sum;
    }
}

// Switch with 5+ cases
__global__ void switch_many_cases(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = in[tid];
        int result;
        switch (x % 6) {
            case 0: result = 1; break;
            case 1: result = 2; break;
            case 2: result = 4; break;
            case 3: result = 8; break;
            case 4: result = 16; break;
            case 5: result = 32; break;
            default: result = 0; break;
        }
        out[tid] = result;
    }
}

// Switch followed by more code
__global__ void switch_then_use(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int category;
        switch (v % 3) {
            case 0: category = -1; break;
            case 1: category = 0; break;
            default: category = 1; break;
        }
        // Use category in subsequent expression
        out[tid] = v * category + 100;
    }
}
