// Regression: inline PTX assembly asm(...); not handled
// Without fix: 'asm' was parsed as a function call; the ':' separators in
//   asm constraint strings caused ParseError "expected RPAREN, got COLON ':'".
// Fix: _parse_stmt detects 'asm'/'__asm__'/'__asm' and skips to the next ';'
//   by balancing parentheses (result register is left uninitialized — known
//   limitation; the PTX body itself is not emitted).

__global__ void inline_asm_skip(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = tid;
        // This asm statement is silently skipped; 'val' stays as 'tid'.
        asm("add.s32 %0, %0, 1;" : "+r"(val));
        out[tid] = val;
    }
}

__global__ void asm_volatile_test(unsigned int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int clock_val = 0;
        asm volatile("mov.u32 %0, %clock;" : "=r"(clock_val));
        out[tid] = clock_val;
    }
}
