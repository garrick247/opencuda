// Regression: bare block { ... } as a statement
// Without fix: _parse_stmt fell through to expression statement handler;
//   _parse_primary_expr saw '{' → ParseError "unexpected token '{'".
// Fix: _parse_stmt checks for LBRACE at the start and calls _parse_stmt_or_block.

__global__ void bare_block_test(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int x = tid;

        // Bare block creates a nested scope (variables flatten in SSA IR)
        {
            int y = x * 2;
            out[tid] = y;
        }

        // Multiple bare blocks
        {
            int a = x + 1;
            out[tid] += a;
        }
        {
            int b = x - 1;
            out[tid] += b;
        }
    }
}

// Bare block at function scope (not inside if)
__global__ void bare_block_top(float *out, float *in, int n) {
    {
        int tid = threadIdx.x;
        if (tid < n) {
            out[tid] = in[tid] * 2.0f;
        }
    }
}

// Nested bare blocks
__global__ void nested_blocks(int *out, int n) {
    int tid = threadIdx.x;
    {
        int a = tid + 1;
        {
            int b = a * 2;
            {
                if (tid < n) out[tid] = b;
            }
        }
    }
}
