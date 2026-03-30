// Generic runtime validation harness for OpenCUDA.
//
// Loads an OpenCUDA-generated PTX file, JIT compiles it, runs each kernel
// found in the PTX against an nvcc-compiled reference, and compares outputs.
//
// Build:  nvcc -o runtime_validate runtime_validate.cu -lcuda
// Usage:  runtime_validate <opencuda.ptx> [N]
//
// The harness auto-discovers kernels in the PTX by name matching against
// a registry of known test kernels. Each registered kernel provides:
//   - Input generation function
//   - Reference computation (CPU-side, same algorithm)
//   - Output comparison with configurable tolerance

#include <cuda.h>
#include <cuda_runtime.h>
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

// ======================== TEST FRAMEWORK ========================

struct TestResult {
    const char* name;
    int passed;
    int total;
    float max_diff;
};

typedef int (*TestFunc)(CUfunction func, int N, TestResult* result);

struct TestEntry {
    const char* kernel_name;
    TestFunc    test_fn;
};

char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

// ======================== GENERIC HELPERS ========================

// Run an OpenCUDA kernel with (out, a, b, n) signature, compare against CPU ref.
int generic_float3_test(CUfunction func, int N, TestResult* result,
                        void (*init)(float* a, float* b, int n),
                        void (*cpu_ref)(float* out, float* a, float* b, int n),
                        float tol) {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_ref = (float*)malloc(N * sizeof(float));
    float *h_oc = (float*)malloc(N * sizeof(float));

    init(h_a, h_b, N);
    cpu_ref(h_ref, h_a, h_b, N);

    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_out, N * sizeof(float));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(float));
    cuMemcpyHtoD(d_b, h_b, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, N * sizeof(float));

    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_ref[i] - h_oc[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (errors < 3)
                printf("    MISMATCH [%d]: ref=%.6f oc=%.6f diff=%.6f\n",
                       i, h_ref[i], h_oc[i], diff);
            errors++;
        }
    }

    result->total = N;
    result->passed = N - errors;
    result->max_diff = max_diff;

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_b); free(h_ref); free(h_oc);
    return errors;
}

// ======================== TEST KERNELS ========================

// --- vector_add: out = a + b ---
void vadd_init(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) { a[i] = (float)(i % 100) * 0.1f; b[i] = (float)(i % 37) * 0.3f - 5.0f; }
}
void vadd_ref(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}
int test_vector_add(CUfunction f, int N, TestResult* r) {
    r->name = "vector_add";
    return generic_float3_test(f, N, r, vadd_init, vadd_ref, 0.001f);
}

// --- saxpy: out[i] = a[i] * b[i] (element-wise multiply as proxy for saxpy) ---
void emul_init(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) { a[i] = (float)(i % 50) * 0.02f; b[i] = (float)(i % 23) - 11.0f; }
}

// --- reduce: out = sum(in) using block reduction ---
int test_reduce(CUfunction func, int N, TestResult* r) {
    r->name = "reduce_sum";
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 17) * 0.1f - 0.8f;

    // CPU reference
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) cpu_sum += (double)h_in[i];

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float *h_out = (float*)calloc(blocks, sizeof(float));

    CUdeviceptr d_in, d_out;
    cuMemAlloc(&d_in, N * sizeof(float));
    cuMemAlloc(&d_out, blocks * sizeof(float));
    cuMemcpyHtoD(d_in, h_in, N * sizeof(float));
    cuMemsetD8(d_out, 0, blocks * sizeof(float));

    void* args[] = { &d_out, &d_in, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_out, d_out, blocks * sizeof(float));

    double gpu_sum = 0.0;
    for (int i = 0; i < blocks; i++) gpu_sum += (double)h_out[i];

    float diff = (float)fabs(cpu_sum - gpu_sum);
    float tol = (float)fabs(cpu_sum) * 1e-4f + 1e-4f;  // relative + absolute

    r->total = 1;
    r->passed = (diff <= tol) ? 1 : 0;
    r->max_diff = diff;

    if (r->passed)
        printf("    cpu_sum=%.6f gpu_sum=%.6f diff=%.9f\n", (float)cpu_sum, (float)gpu_sum, diff);
    else
        printf("    FAIL: cpu_sum=%.6f gpu_sum=%.6f diff=%.6f (tol=%.6f)\n",
               (float)cpu_sum, (float)gpu_sum, diff, tol);

    cuMemFree(d_in); cuMemFree(d_out);
    free(h_in); free(h_out);
    return r->passed ? 0 : 1;
}

// --- generic: out = a * b ---
void vmul_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] * b[i]; }
int test_vector_mul(CUfunction f, int N, TestResult* r) {
    r->name = "vector_mul";
    return generic_float3_test(f, N, r, vadd_init, vmul_ref, 0.001f);
}

// --- generic: out = -a ---
void vneg_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = -a[i]; }
int test_vector_neg(CUfunction f, int N, TestResult* r) {
    r->name = "vector_neg";
    return generic_float3_test(f, N, r, vadd_init, vneg_ref, 0.001f);
}

// --- generic: out = a * a ---
void vsq_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] * a[i]; }
int test_vector_sq(CUfunction f, int N, TestResult* r) {
    r->name = "vector_sq";
    return generic_float3_test(f, N, r, vadd_init, vsq_ref, 0.001f);
}

// --- generic: out = a * b + a ---
void vfma_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] * b[i] + a[i]; }
int test_vector_fma(CUfunction f, int N, TestResult* r) {
    r->name = "vector_fma";
    return generic_float3_test(f, N, r, vadd_init, vfma_ref, 0.001f);
}

// --- generic: out = max(a, b) ---
void vmax_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] > b[i] ? a[i] : b[i]; }
int test_vector_max(CUfunction f, int N, TestResult* r) {
    r->name = "vector_max";
    return generic_float3_test(f, N, r, vadd_init, vmax_ref, 0.001f);
}

// --- generic: out = abs(a) ---
void vabs_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] < 0 ? -a[i] : a[i]; }
int test_vector_abs(CUfunction f, int N, TestResult* r) {
    r->name = "vector_abs";
    return generic_float3_test(f, N, r, vadd_init, vabs_ref, 0.001f);
}

// --- generic: out = clamp(a, 0, 1) ---
void vclamp_ref(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) { float v = a[i]; out[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
}
int test_vector_clamp01(CUfunction f, int N, TestResult* r) {
    r->name = "vector_clamp01";
    return generic_float3_test(f, N, r, vadd_init, vclamp_ref, 0.001f);
}

// --- block_reverse: reverse within each 256-element block ---
void brev_ref(float* out, float* a, float* b, int n) {
    // Reverse within blocks of 256
    for (int blk = 0; blk < n; blk += 256) {
        int end = (blk + 256 < n) ? blk + 256 : n;
        int sz = end - blk;
        for (int i = 0; i < sz; i++) out[blk + sz - 1 - i] = a[blk + i];
    }
}
int test_block_reverse(CUfunction f, int N, TestResult* r) {
    r->name = "block_reverse";
    return generic_float3_test(f, N, r, vadd_init, brev_ref, 0.001f);
}

// --- warp_reduce: sum of each 32-element warp → out[warp_id] ---
int test_warp_reduce(CUfunction func, int N, TestResult* r) {
    r->name = "warp_reduce";
    float *h_a = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_a[i] = (float)(i % 100) * 0.1f;

    int n_warps = (N + 31) / 32;
    float *h_ref = (float*)calloc(n_warps, sizeof(float));
    for (int i = 0; i < N; i++) h_ref[i / 32] += h_a[i];

    float *h_oc = (float*)calloc(n_warps, sizeof(float));
    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));  // unused but needed for signature
    cuMemAlloc(&d_out, n_warps * sizeof(float));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(float));
    cuMemsetD8(d_out, 0, n_warps * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, n_warps * sizeof(float));

    int errors = 0;
    float max_diff = 0.0f;
    for (int w = 0; w < n_warps; w++) {
        float diff = fabsf(h_ref[w] - h_oc[w]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) { errors++; if (errors <= 3) printf("    MISMATCH warp %d: ref=%.3f oc=%.3f\n", w, h_ref[w], h_oc[w]); }
    }
    r->total = n_warps; r->passed = n_warps - errors; r->max_diff = max_diff;
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_ref); free(h_oc);
    return errors;
}

// --- prefix_sum: inclusive scan within each block ---
int test_prefix_sum(CUfunction func, int N, TestResult* r) {
    r->name = "prefix_sum";
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_ref = (float*)malloc(N * sizeof(float));
    float *h_oc = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_a[i] = (float)((i % 10) - 5) * 0.1f;
    // CPU: prefix sum within blocks of 256
    for (int blk = 0; blk < N; blk += 256) {
        float s = 0.0f;
        int end = (blk + 256 < N) ? blk + 256 : N;
        for (int i = blk; i < end; i++) { s += h_a[i]; h_ref[i] = s; }
    }

    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(float));
    cuMemAlloc(&d_b, N * sizeof(float));
    cuMemAlloc(&d_out, N * sizeof(float));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(float));

    int threads = 256, blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, N * sizeof(float));

    int errors = 0; float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_ref[i] - h_oc[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.01f) { errors++; if (errors <= 3) printf("    MISMATCH [%d]: ref=%.4f oc=%.4f\n", i, h_ref[i], h_oc[i]); }
    }
    r->total = N; r->passed = N - errors; r->max_diff = max_diff;
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_ref); free(h_oc);
    return errors;
}

// --- relu: max(0, x) ---
void relu_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] > 0 ? a[i] : 0.0f; }
int test_relu(CUfunction f, int N, TestResult* r) {
    r->name = "relu";
    return generic_float3_test(f, N, r, vadd_init, relu_ref, 0.001f);
}

// --- sigmoid: 1/(1+exp(-x)) ---
void sigmoid_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = 1.0f / (1.0f + expf(-a[i])); }
int test_sigmoid(CUfunction f, int N, TestResult* r) {
    r->name = "sigmoid";
    return generic_float3_test(f, N, r, vadd_init, sigmoid_ref, 0.001f);
}

// --- stencil_1d: 0.25*a[i-1] + 0.5*a[i] + 0.25*a[i+1] ---
void stencil_ref(float* out, float* a, float* b, int n) {
    for (int blk = 0; blk < n; blk += 256) {
        int end = (blk + 256 < n) ? blk + 256 : n;
        for (int i = blk; i < end; i++) {
            float left  = (i > 0) ? a[i-1] : 0.0f;
            float right = (i < n-1) ? a[i+1] : 0.0f;
            out[i] = 0.25f * left + 0.5f * a[i] + 0.25f * right;
        }
    }
}
int test_stencil(CUfunction f, int N, TestResult* r) {
    r->name = "stencil_1d";
    return generic_float3_test(f, N, r, vadd_init, stencil_ref, 0.01f);
}

// ======================== INTEGER TESTS ========================

// Generic int test: (int *out, int *a, int *b, int n)
int generic_int3_test(CUfunction func, int N, TestResult* result,
                      void (*init)(int* a, int* b, int n),
                      void (*cpu_ref)(int* out, int* a, int* b, int n)) {
    int *h_a = (int*)malloc(N * sizeof(int));
    int *h_b = (int*)malloc(N * sizeof(int));
    int *h_ref = (int*)malloc(N * sizeof(int));
    int *h_oc = (int*)malloc(N * sizeof(int));

    init(h_a, h_b, N);
    cpu_ref(h_ref, h_a, h_b, N);

    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(int));
    cuMemAlloc(&d_b, N * sizeof(int));
    cuMemAlloc(&d_out, N * sizeof(int));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(int));
    cuMemcpyHtoD(d_b, h_b, N * sizeof(int));

    int threads = 256, blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, N * sizeof(int));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_ref[i] != h_oc[i]) {
            if (errors < 3) printf("    MISMATCH [%d]: ref=%d oc=%d\n", i, h_ref[i], h_oc[i]);
            errors++;
        }
    }
    result->total = N; result->passed = N - errors; result->max_diff = (float)errors;
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_b); free(h_ref); free(h_oc);
    return errors;
}

void iinit(int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { a[i] = (i * 73 + 17) % 1000 - 500; b[i] = (i * 37 + 7) % 100; }
}

void iadd_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] + b[i]; }
int test_int_add(CUfunction f, int N, TestResult* r) { r->name = "int_add"; return generic_int3_test(f, N, r, iinit, iadd_ref); }

void imul_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] * b[i]; }
int test_int_mul(CUfunction f, int N, TestResult* r) { r->name = "int_mul"; return generic_int3_test(f, N, r, iinit, imul_ref); }

void ixor_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] ^ b[i]; }
int test_int_xor(CUfunction f, int N, TestResult* r) { r->name = "int_xor"; return generic_int3_test(f, N, r, iinit, ixor_ref); }

void iand_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] & b[i]; }
int test_int_and(CUfunction f, int N, TestResult* r) { r->name = "int_and"; return generic_int3_test(f, N, r, iinit, iand_ref); }

void ishl_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] << (b[i] & 31); }
int test_int_shl(CUfunction f, int N, TestResult* r) { r->name = "int_shl"; return generic_int3_test(f, N, r, iinit, ishl_ref); }

void imax_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = a[i] > b[i] ? a[i] : b[i]; }
int test_int_max(CUfunction f, int N, TestResult* r) { r->name = "int_max"; return generic_int3_test(f, N, r, iinit, imax_ref); }

void iabsdiff_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) { int d = a[i]-b[i]; o[i] = d < 0 ? -d : d; } }
int test_int_absdiff(CUfunction f, int N, TestResult* r) { r->name = "int_absdiff"; return generic_int3_test(f, N, r, iinit, iabsdiff_ref); }

// popcount CPU reference
int cpu_popc(int v) { unsigned u = (unsigned)v; int c = 0; while (u) { c += u & 1; u >>= 1; } return c; }
void ipopc_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = cpu_popc(a[i]); }
int test_int_popc(CUfunction f, int N, TestResult* r) { r->name = "int_popc"; return generic_int3_test(f, N, r, iinit, ipopc_ref); }

// clz CPU reference
int cpu_clz(int v) { unsigned u = (unsigned)v; if (u == 0) return 32; int c = 0; while (!(u & 0x80000000u)) { c++; u <<= 1; } return c; }
void iclz_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = cpu_clz(a[i]); }
int test_int_clz(CUfunction f, int N, TestResult* r) { r->name = "int_clz"; return generic_int3_test(f, N, r, iinit, iclz_ref); }

// brev CPU reference
unsigned cpu_brev(unsigned v) { unsigned r = 0; for (int i = 0; i < 32; i++) { r <<= 1; r |= v & 1; v >>= 1; } return r; }
void ibrev_ref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = (int)cpu_brev((unsigned)a[i]); }
int test_int_brev(CUfunction f, int N, TestResult* r) { r->name = "int_brev"; return generic_int3_test(f, N, r, iinit, ibrev_ref); }

// ======================== ADVANCED TESTS ========================

// lerp: out = a + (b - a) * 0.3
void lerp_ref(float* out, float* a, float* b, int n) { for (int i = 0; i < n; i++) out[i] = a[i] + (b[i] - a[i]) * 0.3f; }
int test_lerp(CUfunction f, int N, TestResult* r) { r->name = "lerp_kernel"; return generic_float3_test(f, N, r, vadd_init, lerp_ref, 0.001f); }

// classify: ternary chain
void classify_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { int v = a[i]; o[i] = v < -10 ? -2 : v < 0 ? -1 : v == 0 ? 0 : v < 10 ? 1 : 2; }
}
int test_classify(CUfunction f, int N, TestResult* r) { r->name = "classify"; return generic_int3_test(f, N, r, iinit, classify_iref); }

// cond_accum: sum positive values in sliding window of 16
void cond_accum_iref(int* o, int* a, int* b, int n) {
    for (int g = 0; g < n; g++) { int s = 0; for (int i = 0; i < 16; i++) { int v = a[(g+i) % n]; if (v > 0) s += v; } o[g] = s; }
}
int test_cond_accum(CUfunction f, int N, TestResult* r) { r->name = "cond_accum"; return generic_int3_test(f, N, r, iinit, cond_accum_iref); }

// float_to_int: (int)(a * 0.1 + 0.5)
void f2i_iref(int* o, int* a, int* b, int n) { for (int i = 0; i < n; i++) o[i] = (int)((float)a[i] * 0.1f + 0.5f); }
int test_float_to_int(CUfunction f, int N, TestResult* r) { r->name = "float_to_int"; return generic_int3_test(f, N, r, iinit, f2i_iref); }

// divmod: (a/b)*1000 + (a%b)
void divmod_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { int d = b[i] != 0 ? b[i] : 1; o[i] = (a[i] / d) * 1000 + (a[i] % d); }
}
int test_divmod(CUfunction f, int N, TestResult* r) { r->name = "divmod"; return generic_int3_test(f, N, r, iinit, divmod_iref); }

// warp_scan: inclusive prefix sum per warp
int test_warp_scan(CUfunction func, int N, TestResult* r) {
    r->name = "warp_scan";
    int *h_a = (int*)malloc(N * sizeof(int));
    int *h_ref = (int*)malloc(N * sizeof(int));
    int *h_oc = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_a[i] = (i % 5) - 2;
    // CPU: inclusive prefix sum within warps of 32
    for (int w = 0; w < N; w += 32) {
        int s = 0;
        int end = (w + 32 < N) ? w + 32 : N;
        for (int i = w; i < end; i++) { s += h_a[i]; h_ref[i] = s; }
    }
    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(int));
    cuMemAlloc(&d_b, N * sizeof(int));
    cuMemAlloc(&d_out, N * sizeof(int));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(int));
    int threads = 256, blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, N * sizeof(int));
    int errors = 0;
    for (int i = 0; i < N; i++) if (h_ref[i] != h_oc[i]) { errors++; if (errors <= 3) printf("    MISMATCH [%d]: ref=%d oc=%d\n", i, h_ref[i], h_oc[i]); }
    r->total = N; r->passed = N - errors; r->max_diff = (float)errors;
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_ref); free(h_oc);
    return errors;
}

// sum_and_diff: s*1000 + abs(d)
void sd_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { int s = a[i]+b[i]; int d = a[i]-b[i]; if (d<0) d=-d; o[i] = s*1000+d; }
}
int test_sum_and_diff(CUfunction f, int N, TestResult* r) { r->name = "sum_and_diff"; return generic_int3_test(f, N, r, iinit, sd_iref); }

// ======================== RECURSIVE TESTS ========================

int cpu_factorial(int n) { return (n <= 1) ? 1 : n * cpu_factorial(n-1); }
void factorial_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { int v = a[i] % 13; if (v < 0) v = -v; o[i] = cpu_factorial(v); }
}
int test_factorial(CUfunction f, int N, TestResult* r) { r->name = "factorial"; return generic_int3_test(f, N, r, iinit, factorial_iref); }

int cpu_fib(int n) { if (n <= 0) return 0; if (n == 1) return 1; return cpu_fib(n-1) + cpu_fib(n-2); }
void fibonacci_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) { int v = a[i] % 15; if (v < 0) v = -v; o[i] = cpu_fib(v); }
}
int test_fibonacci(CUfunction f, int N, TestResult* r) { r->name = "fibonacci"; return generic_int3_test(f, N, r, iinit, fibonacci_iref); }

int cpu_gcd(int a, int b) { if (b == 0) return a; return cpu_gcd(b, a % b); }
void gcd_iref(int* o, int* a, int* b, int n) {
    for (int i = 0; i < n; i++) {
        int va = a[i]; if (va < 0) va = -va; if (va == 0) va = 1;
        int vb = b[i]; if (vb < 0) vb = -vb; if (vb == 0) vb = 1;
        o[i] = cpu_gcd(va, vb);
    }
}
int test_gcd(CUfunction f, int N, TestResult* r) { r->name = "gcd"; return generic_int3_test(f, N, r, iinit, gcd_iref); }

// ======================== ATOMIC TESTS ========================

// Generic atomic test: runs kernel with (out, a, b, n), out is a single int.
int generic_atomic_int(CUfunction func, int N, TestResult* result,
                       int init_val, int (*cpu_ref)(int* a, int n)) {
    int *h_a = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_a[i] = (i * 73 + 17) % 1000 - 500;
    int expected = cpu_ref(h_a, N);

    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, N * sizeof(int));
    cuMemAlloc(&d_b, N * sizeof(int));
    cuMemAlloc(&d_out, sizeof(int));
    cuMemcpyHtoD(d_a, h_a, N * sizeof(int));
    cuMemcpyHtoD(d_out, &init_val, sizeof(int));

    int threads = 256, blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    int h_out;
    cuMemcpyDtoH(&h_out, d_out, sizeof(int));

    int err = (h_out != expected) ? 1 : 0;
    if (err) printf("    MISMATCH: gpu=%d cpu=%d\n", h_out, expected);
    else printf("    gpu=%d cpu=%d\n", h_out, expected);

    result->total = 1; result->passed = 1 - err; result->max_diff = (float)abs(h_out - expected);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a);
    return err;
}

int cpu_sum(int* a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
int test_atomic_sum(CUfunction f, int N, TestResult* r) { r->name = "atomic_sum"; return generic_atomic_int(f, N, r, 0, cpu_sum); }

int cpu_max(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] > m) m = a[i]; return m; }
int test_atomic_max(CUfunction f, int N, TestResult* r) { r->name = "atomic_max"; return generic_atomic_int(f, N, r, -2147483647, cpu_max); }

int cpu_min(int* a, int n) { int m = a[0]; for (int i = 1; i < n; i++) if (a[i] < m) m = a[i]; return m; }
int test_atomic_min(CUfunction f, int N, TestResult* r) { r->name = "atomic_min"; return generic_atomic_int(f, N, r, 2147483647, cpu_min); }

int cpu_pos_count(int* a, int n) { int c = 0; for (int i = 0; i < n; i++) if (a[i] > 0) c++; return c; }
int test_ballot_count(CUfunction f, int N, TestResult* r) { r->name = "ballot_count"; return generic_atomic_int(f, N, r, 0, cpu_pos_count); }

// --- matmul: C = A * B (square NxN) ---
int test_matmul(CUfunction func, int N, TestResult* r) {
    r->name = "matmul_simple";
    // Use smaller N for matmul (N^2 elements, N^3 ops)
    int dim = (N > 64) ? 64 : N;
    int sz = dim * dim;
    float *h_a = (float*)malloc(sz * sizeof(float));
    float *h_b = (float*)malloc(sz * sizeof(float));
    float *h_ref = (float*)malloc(sz * sizeof(float));
    float *h_oc = (float*)malloc(sz * sizeof(float));

    for (int i = 0; i < sz; i++) { h_a[i] = (float)(i % 7) * 0.1f - 0.3f; h_b[i] = (float)(i % 11) * 0.1f - 0.5f; }
    // CPU matmul
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++) {
            float s = 0.0f;
            for (int k = 0; k < dim; k++) s += h_a[i*dim+k] * h_b[k*dim+j];
            h_ref[i*dim+j] = s;
        }

    CUdeviceptr d_a, d_b, d_out;
    cuMemAlloc(&d_a, sz * sizeof(float));
    cuMemAlloc(&d_b, sz * sizeof(float));
    cuMemAlloc(&d_out, sz * sizeof(float));
    cuMemcpyHtoD(d_a, h_a, sz * sizeof(float));
    cuMemcpyHtoD(d_b, h_b, sz * sizeof(float));

    // 2D launch
    dim3 threads(16, 16);
    dim3 blocks((dim+15)/16, (dim+15)/16);
    void* args[] = { &d_out, &d_a, &d_b, &dim };
    CHECK_CU(cuLaunchKernel(func, blocks.x, blocks.y, 1, threads.x, threads.y, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());
    cuMemcpyDtoH(h_oc, d_out, sz * sizeof(float));

    int errors = 0; float max_diff = 0.0f;
    for (int i = 0; i < sz; i++) {
        float diff = fabsf(h_ref[i] - h_oc[i]);
        if (diff > max_diff) max_diff = diff;
        float tol = fabsf(h_ref[i]) * 1e-4f + 1e-4f;
        if (diff > tol) { errors++; if (errors <= 3) printf("    MISMATCH [%d]: ref=%.6f oc=%.6f diff=%.6f\n", i, h_ref[i], h_oc[i], diff); }
    }
    r->total = sz; r->passed = sz - errors; r->max_diff = max_diff;
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_b); free(h_ref); free(h_oc);
    return errors;
}

// ======================== REGISTRY ========================

TestEntry g_tests[] = {
    { "vector_add",     test_vector_add },
    { "vector_mul",     test_vector_mul },
    { "vector_neg",     test_vector_neg },
    { "vector_sq",      test_vector_sq },
    { "vector_fma",     test_vector_fma },
    { "vector_max",     test_vector_max },
    { "vector_abs",     test_vector_abs },
    { "vector_clamp01", test_vector_clamp01 },
    { "reduce_sum",     test_reduce },
    { "block_reverse",  test_block_reverse },
    { "warp_reduce",    test_warp_reduce },
    { "prefix_sum",     test_prefix_sum },
    { "relu",           test_relu },
    { "sigmoid",        test_sigmoid },
    { "stencil_1d",     test_stencil },
    { "int_add",        test_int_add },
    { "int_mul",        test_int_mul },
    { "int_xor",        test_int_xor },
    { "int_and",        test_int_and },
    { "int_shl",        test_int_shl },
    { "int_max",        test_int_max },
    { "int_absdiff",    test_int_absdiff },
    { "int_popc",       test_int_popc },
    { "int_clz",        test_int_clz },
    { "int_brev",       test_int_brev },
    { "matmul_simple",  test_matmul },
    { "atomic_sum",     test_atomic_sum },
    { "atomic_max_k",   test_atomic_max },
    { "atomic_min_k",   test_atomic_min },
    { "ballot_count",   test_ballot_count },
    { "lerp_kernel",    test_lerp },
    { "classify",       test_classify },
    { "cond_accum",     test_cond_accum },
    { "float_to_int",   test_float_to_int },
    { "divmod",         test_divmod },
    { "warp_scan",      test_warp_scan },
    { "sum_and_diff",   test_sum_and_diff },
    { "factorial_k",    test_factorial },
    { "fibonacci_k",    test_fibonacci },
    { "gcd_k",          test_gcd },
    { NULL, NULL }
};

// ======================== MAIN ========================

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <opencuda_ptx_file> [N]\n", argv[0]);
        fprintf(stderr, "\nAuto-discovers kernels in PTX and validates against CPU reference.\n");
        fprintf(stderr, "Known kernels: ");
        for (int i = 0; g_tests[i].kernel_name; i++)
            fprintf(stderr, "%s ", g_tests[i].kernel_name);
        fprintf(stderr, "\n");
        return 1;
    }
    const char* ptx_path = argv[1];
    int N = (argc > 2) ? atoi(argv[2]) : 1024;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    char devname[256];
    cuDeviceGetName(devname, sizeof(devname), dev);
    printf("Device: %s\n", devname);
    printf("PTX: %s  N=%d\n\n", ptx_path, N);

    char* ptx_source = read_file(ptx_path);
    CUmodule mod;
    CUresult load_err = cuModuleLoadData(&mod, ptx_source);
    if (load_err != CUDA_SUCCESS) {
        const char* str; cuGetErrorString(load_err, &str);
        fprintf(stderr, "JIT load failed: %s\n", str);
        free(ptx_source);
        return 1;
    }
    printf("JIT compilation OK.\n\n");

    int total_pass = 0, total_fail = 0, total_skip = 0;

    for (int i = 0; g_tests[i].kernel_name; i++) {
        CUfunction func;
        CUresult r = cuModuleGetFunction(&func, mod, g_tests[i].kernel_name);
        if (r != CUDA_SUCCESS) {
            total_skip++;
            continue;
        }
        printf("[TEST] %s ...\n", g_tests[i].kernel_name);
        TestResult result = {};
        int err = g_tests[i].test_fn(func, N, &result);
        if (err == 0) {
            printf("  PASS: %d/%d correct, max_diff=%.9f\n\n", result.passed, result.total, result.max_diff);
            total_pass++;
        } else {
            printf("  FAIL: %d/%d correct\n\n", result.passed, result.total);
            total_fail++;
        }
    }

    printf("========================================\n");
    printf("Results: %d PASS, %d FAIL, %d skipped\n", total_pass, total_fail, total_skip);
    printf("========================================\n");

    free(ptx_source);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return total_fail > 0 ? 1 : 0;
}
