// Probe: for-loop body shadows the loop counter itself,
// nested for-loop where inner loop shadows outer loop variable,
// variable in while-body that shadows outer variable,
// for-loop where init shadows a variable from an enclosing for-loop

// Body scope shadows for-loop init variable 'i'
// For each iteration: body's 'i' = constant 5, loop counter still increments
__global__ void body_shadows_counter(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int i = 5;     // body's 'i' shadows loop counter (always 5)
            sum += i;      // sum += 5 per iteration, NOT sum += loop_counter
        }
        // Expected: sum = n * 5
        // The loop ran n iterations (loop counter incremented correctly)
        *out = sum;
    }
}

// Nested for where inner loop reuses outer loop var name
__global__ void nested_same_name(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int total = 0;
        for (int i = 0; i < n; i++) {
            // Inner loop also named 'i', shadows outer i
            for (int i = 0; i < 3; i++) {
                total += i;   // inner i: 0 + 1 + 2 = 3 per outer iteration
            }
            // After inner loop: outer 'i' restored, outer loop continues
        }
        // Expected: total = n * 3
        *out = total;
    }
}

// While-body declares variable shadowing outer
__global__ void while_body_shadow(int *out, int *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int sum = 100;   // outer 'sum'
        int i = 0;
        while (i < n) {
            int sum = in[i];  // body 'sum' shadows outer sum
            // outer sum NOT modified by this loop
            i++;
        }
        // outer sum must still be 100
        *out = sum;
    }
}

// For-loop init that shadows enclosing for's variable
__global__ void shadow_outer_counter(int *out, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        int grand_total = 0;
        for (int i = 0; i < n; i++) {
            int row_sum = 0;
            // Inner for-loop also uses 'i', shadows outer 'i'
            for (int i = 0; i < n; i++) {
                row_sum += i;  // inner i: 0..n-1
            }
            // After inner loop: outer 'i' restored
            grand_total += row_sum + i;  // row_sum + outer_i
        }
        *out = grand_total;
    }
}
