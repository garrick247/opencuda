// Regression: object-like macros referencing other macros not recursively expanded
// Without fix: preprocessor did a single pass over obj_defines by length — shorter
//   names like TOTAL expanded AFTER longer names (MAX_BLOCKS) had already been
//   applied, so TOTAL's expansion body still contained unexpanded WARP/MAX_BLOCKS →
//   ParseError "undefined variable 'MAX_BLOCKS'".
// Fix: preprocess() iterates object-like substitutions up to 8 times until stable.

#define WARP_SIZE  32
#define NUM_WARPS  8
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
#define GRID_SCALE 4
#define TOTAL_THREADS (BLOCK_SIZE * GRID_SCALE)

__global__ void recursive_macro_test(int *out, int n) {
    int tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (tid < n && tid < TOTAL_THREADS) {
        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        out[tid] = warp * NUM_WARPS + lane;
    }
}

// Three-level chain: C → B → A
#define BASE  1
#define MID   (BASE + 1)
#define TOP   (MID * MID)

__global__ void chain_macro(int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = tid + TOP;  // TOP = (BASE+1)*(BASE+1) = 4
    }
}
