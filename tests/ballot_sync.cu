// Regression: __ballot_sync returns a lane bitmask, typed UINT32.
// If typed INT32, operations on the mask that involve signed-vs-unsigned
// mixing could produce wrong types.
__global__ void ballot_test(int *out, int *in, int n) {
    int tid = threadIdx.x;
    int val = (tid < n) ? in[tid] : 0;

    // ballot returns bitmask — must be UINT32 (u32)
    unsigned int active = __ballot_sync(0xFFFFFFFF, val > 0);

    // Bitwise ops on the mask — must use b32 ops
    unsigned int lower = active & 0x0000FFFF;
    unsigned int upper = (active >> 16) & 0x0000FFFF;

    if (tid == 0) {
        out[0] = lower + upper;
    }
}
