// Hardware test harness for first_positive kernel.
// Compiles with: nvcc -o gpu_first_positive gpu_first_positive.cu -lcuda
// Usage: gpu_first_positive <cubin_path> [N]
//
// Tests the kernel: first_positive(float* data, float* out, int n)
// Each thread tid reads data[0..n-1] sequentially and writes the first
// positive value found to out[tid], or 0.0f if none found.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* str; cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, str, err); \
        exit(1); \
    } \
} while(0)

// CPU reference: same logic as the kernel
static float cpu_first_positive(const float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (data[i] > 0.0f) return data[i];
    }
    return 0.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cubin_path> [N]\n", argv[0]);
        return 1;
    }
    const char* cubin_path = argv[1];
    int N = (argc > 2) ? atoi(argv[2]) : 64;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    CUmodule mod;
    CUresult load_err = cuModuleLoad(&mod, cubin_path);
    if (load_err != CUDA_SUCCESS) {
        const char* str; cuGetErrorString(load_err, &str);
        fprintf(stderr, "Failed to load cubin '%s': %s\n", cubin_path, str);
        return 1;
    }
    CUfunction func;
    CUresult func_err = cuModuleGetFunction(&func, mod, "first_positive");
    if (func_err != CUDA_SUCCESS) {
        const char* str; cuGetErrorString(func_err, &str);
        fprintf(stderr, "Failed to find kernel 'first_positive': %s\n", str);
        return 1;
    }

    printf("Loaded %s::first_positive (N=%d)\n", cubin_path, N);

    // Three test cases with different data patterns
    struct TestCase {
        const char* name;
        // data[] will be filled programmatically
    };

    int errors_total = 0;

    // Test case parameters: (first_neg_count, then_pos_value)
    // neg_prefix=0: all positive (first element wins immediately)
    // neg_prefix=N: all negative (should write 0.0)
    // neg_prefix=mid: first positive is data[mid]
    int test_configs[][2] = {
        {0, 5},          // data[0]=5.0 is positive → out[*]=5.0
        {1, 3},          // data[0]=-1, data[1]=3.0 → out[*]=3.0
        {N/2, 7},        // first N/2 negative, then 7.0 → out[*]=7.0
        {N, 0},          // all negative → out[*]=0.0
    };
    int num_tests = sizeof(test_configs) / sizeof(test_configs[0]);

    size_t bytes = N * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    float* h_out  = (float*)malloc(N * sizeof(float));

    CUdeviceptr d_data, d_out;
    CHECK_CU(cuMemAlloc(&d_data, bytes));
    CHECK_CU(cuMemAlloc(&d_out,  N * sizeof(float)));

    for (int tc = 0; tc < num_tests; tc++) {
        int neg_prefix = test_configs[tc][0];
        float pos_val  = (float)test_configs[tc][1];
        const char* label = (neg_prefix == 0)   ? "all_positive_from_zero" :
                            (neg_prefix == N)    ? "all_negative" :
                            (neg_prefix == 1)    ? "one_neg_then_pos" :
                                                   "half_neg_then_pos";

        // Build data array
        for (int i = 0; i < N; i++) {
            if (i < neg_prefix)
                h_data[i] = -(float)(i + 1);   // -1, -2, ...
            else if (i == neg_prefix && neg_prefix < N)
                h_data[i] = pos_val;
            else
                h_data[i] = -(float)(i + 1);   // remaining also negative
        }

        // CPU reference
        float expected = cpu_first_positive(h_data, N);

        // Zero output buffer
        memset(h_out, 0, N * sizeof(float));
        CHECK_CU(cuMemcpyHtoD(d_data, h_data, bytes));
        CHECK_CU(cuMemcpyHtoD(d_out, h_out, N * sizeof(float)));

        // Launch: one block of N threads (N ≤ 1024)
        int threads = N < 1024 ? N : 1024;
        int blocks  = (N + threads - 1) / threads;
        void* args[] = { &d_data, &d_out, &N };
        CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
        CHECK_CU(cuCtxSynchronize());

        CHECK_CU(cuMemcpyDtoH(h_out, d_out, N * sizeof(float)));

        int tc_errors = 0;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_out[i] - expected) > 0.001f) {
                if (tc_errors < 3)
                    printf("  [%s] tid=%d: got %.4f, expected %.4f\n",
                           label, i, h_out[i], expected);
                tc_errors++;
            }
        }
        if (tc_errors == 0) {
            printf("PASS [%s]: all %d threads wrote %.4f\n", label, N, expected);
        } else {
            printf("FAIL [%s]: %d/%d mismatches (expected %.4f)\n",
                   label, tc_errors, N, expected);
            errors_total += tc_errors;
        }
    }

    free(h_data); free(h_out);
    cuMemFree(d_data); cuMemFree(d_out);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);

    if (errors_total == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        printf("\nTOTAL FAILURES: %d\n", errors_total);
        return 1;
    }
}
