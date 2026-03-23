// Nasty: while loop with condition-variable mutation inside the body.
// This is the canonical case that was silently miscompiled before the
// while-loop writeback fix. The condition must re-evaluate updated n.
__global__ void sum_to_n(int* out, int n) {
    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    out[0] = sum;    // should be n*(n-1)/2
}

// Second kernel: while loop with multiple variables mutating each iteration.
__global__ void collatz(int* out, int x) {
    int steps = 0;
    while (x != 1) {
        if (x % 2 == 0) {
            x = x / 2;
        } else {
            x = x * 3 + 1;
        }
        steps = steps + 1;
    }
    out[0] = steps;
}
