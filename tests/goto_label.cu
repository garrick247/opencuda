// Regression: goto statement and labeled statement
// Without fix: ParseError "undefined variable 'label_name'" — the IDENT before ':'
//   was treated as an expression, and goto was an unknown identifier.
// Fix: _parse_stmt handles 'goto ident;' as no-op and 'ident:' label as no-op.
//   (Real goto CFG is not supported — goto expresses early-exit patterns
//    that should be written with break/continue/if instead.)

__global__ void goto_skip(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int v = in[tid];
        int count = 0;
        loop_start:
        if (v <= 1 || count >= 100) goto loop_end;
        v = (v % 2 == 0) ? v / 2 : v * 3 + 1;
        count++;
        goto loop_start;
        loop_end:
        out[tid] = count;
    }
}

// Label with no goto (just marks a position — should parse cleanly)
__global__ void label_only(int *out, int *in, int n) {
    int tid = threadIdx.x;
    prologue:
    if (tid >= n) return;
    int v = in[tid];
    compute:
    v = v * 2 + 1;
    out[tid] = v;
}
