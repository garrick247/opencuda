// End-to-end MLP host: loads OpenCUDA PTX, runs 2-layer network, verifies vs CPU.
// Build: nvcc -o e2e_mlp_host e2e_mlp_host.cu -lcuda
// Usage: e2e_mlp_host e2e_mlp_demo.ptx

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* str; cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, str); \
        exit(1); \
    } \
} while(0)

char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1); fread(buf, 1, sz, f); buf[sz] = '\0'; fclose(f);
    return buf;
}

float prand(int* s) { *s = (*s * 1103515245 + 12345) & 0x7fffffff; return (float)*s / (float)0x7fffffff * 2.0f - 1.0f; }

void cpu_linear(float* out, float* in, float* W, float* b, int batch, int id, int od) {
    for (int r = 0; r < batch; r++)
        for (int c = 0; c < od; c++) {
            float s = 0; for (int k = 0; k < id; k++) s += in[r*id+k] * W[c*id+k];
            out[r*od+c] = s + b[c];
        }
}
void cpu_relu(float* d, int n) { for (int i = 0; i < n; i++) if (d[i] < 0) d[i] = 0; }
int cpu_argmax(float* r, int c) { int b=0; float v=r[0]; for(int i=1;i<c;i++) if(r[i]>v){v=r[i];b=i;} return b; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <ptx>\n", argv[0]); return 1; }

    const int B=8, I=16, H=32, O=4;
    float w1[512], b1[32], w2[128], b2[4], input[B*I];
    int seed = 42;
    for (int i = 0; i < 512; i++) w1[i] = prand(&seed) * 0.5f;
    for (int i = 0; i < 32; i++)  b1[i] = prand(&seed) * 0.1f;
    for (int i = 0; i < 128; i++) w2[i] = prand(&seed) * 0.5f;
    for (int i = 0; i < 4; i++)   b2[i] = prand(&seed) * 0.1f;
    for (int i = 0; i < B*I; i++) input[i] = prand(&seed);

    // CPU ref
    float ch[B*H], cl[B*O]; int cp[B];
    cpu_linear(ch, input, w1, b1, B, I, H);
    cpu_relu(ch, B*H);
    cpu_linear(cl, ch, w2, b2, B, H, O);
    for (int r = 0; r < B; r++) cp[r] = cpu_argmax(&cl[r*O], O);
    printf("CPU: "); for (int r=0;r<B;r++) printf("%d ",cp[r]); printf("\n");

    // GPU
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));
    char dn[256]; cuDeviceGetName(dn, 256, dev);
    printf("GPU: %s\n", dn);

    char* ptx = read_file(argv[1]);
    CUmodule mod; CHECK_CU(cuModuleLoadData(&mod, ptx));
    printf("JIT OK\n");

    CUfunction fn1, fnr, fn2, fna;
    CHECK_CU(cuModuleGetFunction(&fn1, mod, "linear1"));
    CHECK_CU(cuModuleGetFunction(&fnr, mod, "relu_inplace"));
    CHECK_CU(cuModuleGetFunction(&fn2, mod, "linear2"));
    CHECK_CU(cuModuleGetFunction(&fna, mod, "argmax_row"));

    // Alloc + upload
    CUdeviceptr dI, dH, dL, dP, dW1, dB1, dW2, dB2;
    CHECK_CU(cuMemAlloc(&dI, B*I*4)); CHECK_CU(cuMemAlloc(&dH, B*H*4));
    CHECK_CU(cuMemAlloc(&dL, B*O*4)); CHECK_CU(cuMemAlloc(&dP, B*4));
    CHECK_CU(cuMemAlloc(&dW1, 512*4)); CHECK_CU(cuMemAlloc(&dB1, 32*4));
    CHECK_CU(cuMemAlloc(&dW2, 128*4)); CHECK_CU(cuMemAlloc(&dB2, 4*4));
    CHECK_CU(cuMemcpyHtoD(dI, input, B*I*4));
    CHECK_CU(cuMemcpyHtoD(dW1, w1, 512*4)); CHECK_CU(cuMemcpyHtoD(dB1, b1, 32*4));
    CHECK_CU(cuMemcpyHtoD(dW2, w2, 128*4)); CHECK_CU(cuMemcpyHtoD(dB2, b2, 4*4));

    // Pipeline: linear1 → relu → linear2 → argmax
    int batch=B, idim=I, hdim=H, odim=O;
    void* a1[] = { &dH, &dI, &dW1, &dB1, &batch, &idim, &hdim };
    CHECK_CU(cuLaunchKernel(fn1, B,1,1, H,1,1, 0,0, a1, NULL));

    int rn = B*H, rt=256, rb=(rn+255)/256;
    void* ar[] = { &dH, &rn };
    CHECK_CU(cuLaunchKernel(fnr, rb,1,1, rt,1,1, 0,0, ar, NULL));

    void* a2[] = { &dL, &dH, &dW2, &dB2, &batch, &hdim, &odim };
    CHECK_CU(cuLaunchKernel(fn2, B,1,1, O,1,1, 0,0, a2, NULL));

    int rows=B, cols=O;
    void* aa[] = { &dP, &dL, &cols, &rows };
    CHECK_CU(cuLaunchKernel(fna, 1,1,1, B,1,1, 0,0, aa, NULL));
    CHECK_CU(cuCtxSynchronize());

    int gp[B]; float gl[B*O];
    CHECK_CU(cuMemcpyDtoH(gp, dP, B*4));
    CHECK_CU(cuMemcpyDtoH(gl, dL, B*O*4));
    printf("GPU: "); for (int r=0;r<B;r++) printf("%d ",gp[r]); printf("\n\n");

    int match = 0;
    for (int r = 0; r < B; r++) {
        int ok = gp[r] == cp[r]; match += ok;
        printf("  [%d] CPU=%d GPU=%d logits=[%.2f,%.2f,%.2f,%.2f] %s\n",
               r, cp[r], gp[r], gl[r*O], gl[r*O+1], gl[r*O+2], gl[r*O+3],
               ok ? "PASS" : "FAIL");
    }
    printf("\n=== %d/%d predictions match ===\n", match, B);
    printf("%s\n", match == B ? "END-TO-END PASS" : "END-TO-END FAIL");

    cuMemFree(dI); cuMemFree(dH); cuMemFree(dL); cuMemFree(dP);
    cuMemFree(dW1); cuMemFree(dB1); cuMemFree(dW2); cuMemFree(dB2);
    cuModuleUnload(mod); free(ptx); cuDevicePrimaryCtxRelease(dev);
    return match == B ? 0 : 1;
}
