// Probe: Loop unrolling edge cases
// - Trip count exactly = max_unroll (16)
// - Trip count = 1 (trivial unroll)
// - Non-standard starting index with computed trip count
// - Unroll with multiple carried variables of different types
// - Loop where body has conditional (should unroll but not fold condition)

__global__ void unroll_trip16(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 16; i++) {  // exactly max_unroll
            sum += i * i;
        }
        // 0+1+4+9+16+25+36+49+64+81+100+121+144+169+196+225 = 1240
        out[0] = sum;
    }
}

__global__ void unroll_trip1(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 1; i++) {  // trip count 1
            sum = i + 42;
        }
        out[0] = sum;  // 42
    }
}

// Starting at non-zero, trip = 5
__global__ void unroll_from3(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int prod = 1;
        for (int i = 3; i < 8; i++) {  // i = 3,4,5,6,7
            prod *= i;
        }
        out[0] = prod;  // 3*4*5*6*7 = 2520
    }
}

// Two carried variables: sum and product
__global__ void unroll_two_carry(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        int prod = 1;
        for (int i = 1; i <= 5; i++) {
            sum += i;
            prod *= i;
        }
        out[0] = sum;   // 15
        out[1] = prod;  // 120
    }
}

// Loop body has conditional but still unrollable (condition is on i)
__global__ void unroll_with_cond(int *out) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            if (i % 2 == 0) {
                sum += i;
            } else {
                sum -= i;
            }
        }
        // 0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 = -4
        out[0] = sum;
    }
}
