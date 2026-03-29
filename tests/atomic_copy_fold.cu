// Tests that identity_fold correctly propagates copies into atomic args.
// Without the fix: the copy instruction (result = val + 0) gets deleted
// but atom.add still references the dead Value.
__global__ void atomic_copy_fold(int *counter, int *data, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = data[tid];
        // The 'result = val + 0' pattern (written by the parser for
        // certain code paths) must be propagated into the atomic arg.
        int result = val;  // Parser may emit: result = val + 0
        atomicAdd(counter, result);
    }
}
