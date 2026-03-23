// Nasty: nested loops, inner break, outer continue after inner exits.
// Tests that break_targets stack is balanced across two nesting levels.
__global__ void nested_search(int* haystack, int* result, int rows, int cols, int target) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= rows) return;

    int found_col = -1;
    for (int r = 0; r < rows; r++) {
        int row_hit = 0;
        for (int c = 0; c < cols; c++) {
            int val = haystack[r * cols + c];
            if (val == target) {
                found_col = c;
                row_hit = 1;
                break;           // exits inner loop only
            }
        }
        if (row_hit) continue;   // outer continue — skip to next r
        found_col = -2;          // only reached when row had no hit
    }
    result[tid] = found_col;
}
