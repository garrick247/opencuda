// Regression: atomicCAS(addr, compare, val) requires 3 args in PTX:
//   atom.global.cas.b32 dest, [addr], compare, val;
// Without fix: only args[1] was emitted, producing:
//   atom.global.cas.b32 dest, [addr], compare;  (missing 'val')
__global__ void atomic_cas_test(int *lock, int *out, int tid_in) {
    // Spin-lock acquire pattern: CAS(lock, 0, 1)
    int old = atomicCAS(lock, 0, 1);
    if (old == 0) {
        // Got the lock
        *out = tid_in;
        atomicExch(lock, 0);  // release
    }
}
