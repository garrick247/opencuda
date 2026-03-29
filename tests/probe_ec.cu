// Probe: C-style variadic macros and unusual identifier forms
// - Identifiers with all digits after _ : _1, __2
// - Identifiers starting with number (should fail)
// - Very long identifier (64+ chars)
// - Macro function with 0 args: FUNC()

#define NOOP() do {} while(0)
#define ZERO_INIT(x) (x) = 0

__global__ void identifier_stress(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int _1 = tid;
        int __2 = tid * 2;
        int very_long_variable_name_that_is_quite_descriptive_and_detailed = tid + _1 + __2;
        NOOP();
        ZERO_INIT(__2);
        out[tid] = very_long_variable_name_that_is_quite_descriptive_and_detailed + __2;
    }
}

// Multiple nesting of macro calls
#define ADD1(x) ((x) + 1)
#define ADD2(x) ADD1(ADD1(x))
#define ADD4(x) ADD2(ADD2(x))

__global__ void nested_macro(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = ADD4(tid);  // should be tid + 4
    }
}
