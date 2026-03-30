// Performance benchmark: compare OpenCUDA PTX vs nvcc-compiled kernels.
// Build: nvcc -O3 -o bench_host bench_host.cu -lcuda
// Usage: bench_host bench_kernels.ptx

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CU(call) do { CUresult e=(call); if(e){const char*s;cuGetErrorString(e,&s);fprintf(stderr,"CU %s:%d %s\n",__FILE__,__LINE__,s);exit(1);}} while(0)
#define CHECK_RT(call) do { cudaError_t e=(call); if(e){fprintf(stderr,"RT %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}} while(0)

char* read_file(const char* p){FILE*f=fopen(p,"rb");if(!f){exit(1);}fseek(f,0,2);long s=ftell(f);fseek(f,0,0);char*b=(char*)malloc(s+1);fread(b,1,s,f);b[s]=0;fclose(f);return b;}

// nvcc-compiled reference kernels
__global__ void ref_vadd(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = a[gid] + b[gid];
}

__global__ void ref_saxpy(float *out, float alpha, float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = alpha * x[gid] + y[gid];
}

__global__ void ref_reduce(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = smem[0];
}

float time_kernel_ms(void (*launch)(float*, float*, float*, int, int), float* out, float* a, float* b, int N, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    // Warmup
    launch(out, a, b, N, 3);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    launch(out, a, b, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

void launch_vadd_ref(float* o, float* a, float* b, int N, int iters) {
    int t=256, bl=(N+t-1)/t;
    for(int i=0;i<iters;i++) ref_vadd<<<bl,t>>>(o,a,b,N);
}
void launch_saxpy_ref(float* o, float* a, float* b, int N, int iters) {
    int t=256, bl=(N+t-1)/t; float alpha=0.5f;
    for(int i=0;i<iters;i++) ref_saxpy<<<bl,t>>>(o,alpha,a,b,N);
}
void launch_reduce_ref(float* o, float* a, float* b, int N, int iters) {
    int t=256, bl=(N+t-1)/t;
    for(int i=0;i<iters;i++) ref_reduce<<<bl,t>>>(o,a,N);
}

float time_cu_kernel(CUfunction fn, CUdeviceptr out, CUdeviceptr a, CUdeviceptr b, int N, int iters, int is_saxpy, int is_reduce) {
    int threads=256, blocks=(N+threads-1)/threads;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    for(int i=0;i<3;i++){
        if (is_saxpy) { float alpha=0.5f; void* args[]={&out,&alpha,&a,&b,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
        else if (is_reduce) { void* args[]={&out,&a,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
        else { void* args[]={&out,&a,&b,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
    }
    cuCtxSynchronize();

    cudaEventRecord(start);
    for(int i=0;i<iters;i++){
        if (is_saxpy) { float alpha=0.5f; void* args[]={&out,&alpha,&a,&b,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
        else if (is_reduce) { void* args[]={&out,&a,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
        else { void* args[]={&out,&a,&b,&N}; cuLaunchKernel(fn,blocks,1,1,threads,1,1,0,0,args,NULL); }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iters;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <opencuda.ptx>\n", argv[0]); return 1; }
    const int N = 1 << 20;  // 1M elements
    const int ITERS = 100;

    // Setup
    float *d_a, *d_b, *d_out;
    CHECK_RT(cudaMalloc(&d_a, N*4)); CHECK_RT(cudaMalloc(&d_b, N*4)); CHECK_RT(cudaMalloc(&d_out, N*4));
    float* h = (float*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=(float)(i%1000)*0.001f;
    CHECK_RT(cudaMemcpy(d_a, h, N*4, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_b, h, N*4, cudaMemcpyHostToDevice));

    char dn[256]; cudaGetDeviceProperties(NULL, 0); // just init
    CHECK_CU(cuInit(0)); CUdevice dev; cuDeviceGet(&dev,0);
    cuDeviceGetName(dn,256,dev);
    printf("Device: %s\n", dn);
    printf("N = %d (%d MB), %d iterations\n\n", N, N*4/1024/1024, ITERS);

    // Load OpenCUDA PTX
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx,dev); cuCtxSetCurrent(ctx);
    char* ptx = read_file(argv[1]);
    CUmodule mod; CHECK_CU(cuModuleLoadData(&mod, ptx));
    CUfunction oc_vadd, oc_saxpy, oc_reduce;
    CHECK_CU(cuModuleGetFunction(&oc_vadd, mod, "bench_vadd"));
    CHECK_CU(cuModuleGetFunction(&oc_saxpy, mod, "bench_saxpy"));
    CHECK_CU(cuModuleGetFunction(&oc_reduce, mod, "bench_reduce"));

    CUdeviceptr cu_a=(CUdeviceptr)d_a, cu_b=(CUdeviceptr)d_b, cu_out=(CUdeviceptr)d_out;

    printf("%-15s %10s %10s %10s\n", "Kernel", "nvcc(us)", "OpenCUDA(us)", "Ratio");
    printf("%-15s %10s %10s %10s\n", "------", "--------", "-----------", "-----");

    // vadd
    float t_nvcc = time_kernel_ms(launch_vadd_ref, d_out, d_a, d_b, N, ITERS) * 1000;
    float t_oc = time_cu_kernel(oc_vadd, cu_out, cu_a, cu_b, N, ITERS, 0, 0) * 1000;
    printf("%-15s %10.1f %10.1f %10.2fx\n", "vector_add", t_nvcc, t_oc, t_oc/t_nvcc);

    // saxpy
    t_nvcc = time_kernel_ms(launch_saxpy_ref, d_out, d_a, d_b, N, ITERS) * 1000;
    t_oc = time_cu_kernel(oc_saxpy, cu_out, cu_a, cu_b, N, ITERS, 1, 0) * 1000;
    printf("%-15s %10.1f %10.1f %10.2fx\n", "saxpy", t_nvcc, t_oc, t_oc/t_nvcc);

    // reduce
    t_nvcc = time_kernel_ms(launch_reduce_ref, d_out, d_a, d_b, N, ITERS) * 1000;
    t_oc = time_cu_kernel(oc_reduce, cu_out, cu_a, cu_b, N, ITERS, 0, 1) * 1000;
    printf("%-15s %10.1f %10.1f %10.2fx\n", "reduce_sum", t_nvcc, t_oc, t_oc/t_nvcc);

    printf("\nRatio = OpenCUDA/nvcc (lower is better, 1.0x = parity)\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); free(h); free(ptx);
    cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);
    return 0;
}
