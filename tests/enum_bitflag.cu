// Regression: enum values referencing other enum names (e.g. EVENT_RW = READ|WRITE)
// Without fix 1: _parse_primary_expr referenced self._lazy_params unconditionally
//   → AttributeError at module level (outside a kernel).
// Without fix 2: _const_fold used BinOp.REM (nonexistent) instead of BinOp.MOD
//   → AttributeError silently swallowed by try/except → _const_fold returned None
//   → _new_val called with self._kernel=None → AttributeError crash.
// Fix: hasattr guard for _lazy_params; _const_fold uses correct BinOp.MOD;
//   constant-folding added to all binary expression parsers (|, &, ^, +, -, *, /, %).

enum EventFlags {
    EVENT_NONE  = 0,
    EVENT_READ  = 1,
    EVENT_WRITE = 2,
    EVENT_RW    = EVENT_READ | EVENT_WRITE,    // cross-reference
    EVENT_ERROR = 4,
    EVENT_ALL   = EVENT_RW | EVENT_ERROR       // nested cross-reference
};

enum Masks {
    MASK_LOW  = 0xFF,
    MASK_HIGH = MASK_LOW << 8,    // shift
    MASK_BOTH = MASK_LOW | MASK_HIGH
};

__global__ void bitflag_test(int *out, int *events, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int ev = events[tid];
        int is_rw    = (ev & EVENT_RW)    != EVENT_NONE;
        int has_err  = (ev & EVENT_ERROR) != EVENT_NONE;
        int low_byte = ev & MASK_LOW;
        out[tid] = is_rw * 100 + has_err * 10 + (low_byte & 0xF);
    }
}
