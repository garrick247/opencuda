// Regression: __all_sync / __any_sync / __ballot_sync warp vote intrinsics.
// Without fix: __all_sync/__any_sync had no parser handler → fell through to
//   unknown device function call; __ballot_sync emitted integer register as
//   pred arg in vote.sync.ballot.b32 → ptxas "Arguments mismatch for 'vote'".
// Fix: parser handles __all_sync/__any_sync as CallInst with INT32 return;
//   codegen converts integer condition to predicate via setp.ne.s32 (or passes
//   an existing pred directly), then emits vote.sync.{all,any,ballot}.

__global__ void warp_vote_test(int *flags, int *out, int n) {
    int tid = threadIdx.x;
    int lane = tid & 31;

    if (tid < n) {
        int flag = flags[tid];
        unsigned int mask = 0xFFFFFFFF;

        // __all_sync: integer arg (needs setp conversion)
        int all_set = __all_sync(mask, flag);

        // __any_sync: integer arg
        int any_set = __any_sync(mask, flag);

        // __ballot_sync: integer arg
        unsigned int ballot = __ballot_sync(mask, flag);

        out[tid] = all_set + any_set + (int)(ballot >> lane);
    }
}

__global__ void warp_vote_pred(int *in, int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int val = in[tid];
        unsigned int mask = 0xFFFFFFFF;

        // Predicate arg (from comparison — already a pred register, no setp needed)
        int all_pos = __all_sync(mask, val > 0);
        int any_pos = __any_sync(mask, val > 0);
        unsigned int ballot_pos = __ballot_sync(mask, val > 0);

        out[tid] = all_pos + any_pos + (int)ballot_pos;
    }
}
