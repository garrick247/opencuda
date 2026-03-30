// Compute-bound benchmark host: OpenCUDA PTX vs nvcc on ALU-heavy kernels.
// Build: nvcc -O3 -o bench_compute_host bench_compute_host.cu -lcuda
// Usage: bench_compute_host bench_compute.ptx

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CU(call) do{CUresult e=(call);if(e){const char*s;cuGetErrorString(e,&s);fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,s);exit(1);}}while(0)
#define CHECK_RT(call) do{cudaError_t e=(call);if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

char* read_file(const char* p){FILE*f=fopen(p,"rb");if(!f)exit(1);fseek(f,0,2);long s=ftell(f);fseek(f,0,0);char*b=(char*)malloc(s+1);fread(b,1,s,f);b[s]=0;fclose(f);return b;}

// ===== NVCC reference kernels (identical source) =====

__global__ void ref_poly(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float x = a[gid];
        float r = 0.01f;
        r = r * x + 0.02f; r = r * x + 0.03f; r = r * x + 0.04f;
        r = r * x + 0.05f; r = r * x + 0.06f; r = r * x + 0.07f;
        r = r * x + 0.08f; r = r * x + 0.09f; r = r * x + 0.10f;
        r = r * x + 0.11f; r = r * x + 0.12f; r = r * x + 0.13f;
        r = r * x + 0.14f; r = r * x + 0.15f;
        out[gid] = r;
    }
}

__global__ void ref_nbody(float *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float ax = 0.0f, ay = 0.0f;
        float px = a[gid], py = b[gid];
        for (int j = 0; j < 64; j++) {
            float dx = a[j % n] - px;
            float dy = b[j % n] - py;
            float r2 = dx*dx + dy*dy + 0.001f;
            float inv = rsqrtf(r2);
            float inv3 = inv * inv * inv;
            ax += dx * inv3;
            ay += dy * inv3;
        }
        out[gid] = ax + ay;
    }
}

__global__ void ref_mandelbrot(int *out, float *a, float *b, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float cr = a[gid] * 3.0f - 2.0f;
        float ci = b[gid] * 2.0f - 1.0f;
        float zr = 0.0f, zi = 0.0f;
        int iter = 0;
        while (zr*zr + zi*zi < 4.0f && iter < 256) {
            float tmp = zr*zr - zi*zi + cr;
            zi = 2.0f*zr*zi + ci;
            zr = tmp;
            iter++;
        }
        out[gid] = iter;
    }
}

// ===== Timing =====

typedef void(*LaunchFn)(void*,void*,void*,int,int);

void launch_ref_poly(void*o,void*a,void*b,int N,int it){int t=256,bl=(N+t-1)/t;for(int i=0;i<it;i++)ref_poly<<<bl,t>>>((float*)o,(float*)a,(float*)b,N);}
void launch_ref_nbody(void*o,void*a,void*b,int N,int it){int t=256,bl=(N+t-1)/t;for(int i=0;i<it;i++)ref_nbody<<<bl,t>>>((float*)o,(float*)a,(float*)b,N);}
void launch_ref_mandel(void*o,void*a,void*b,int N,int it){int t=256,bl=(N+t-1)/t;for(int i=0;i<it;i++)ref_mandelbrot<<<bl,t>>>((int*)o,(float*)a,(float*)b,N);}

float time_ref(LaunchFn fn, void*o,void*a,void*b,int N,int iters){
    fn(o,a,b,N,3); cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s); fn(o,a,b,N,iters); cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms/iters*1000; // microseconds
}

float time_cu(CUfunction fn, CUdeviceptr o, CUdeviceptr a, CUdeviceptr b, int N, int iters, int int_out){
    int t=256, bl=(N+t-1)/t;
    for(int i=0;i<3;i++){void*args[]={&o,&a,&b,&N};cuLaunchKernel(fn,bl,1,1,t,1,1,0,0,args,NULL);}
    cuCtxSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for(int i=0;i<iters;i++){void*args[]={&o,&a,&b,&N};cuLaunchKernel(fn,bl,1,1,t,1,1,0,0,args,NULL);}
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms/iters*1000;
}

int main(int argc, char** argv){
    if(argc<2){fprintf(stderr,"Usage: %s <ptx>\n",argv[0]);return 1;}
    const int N = 1<<20; // 1M elements
    const int ITERS = 50;

    float *d_a, *d_b, *d_fo; int *d_io;
    CHECK_RT(cudaMalloc(&d_a,N*4)); CHECK_RT(cudaMalloc(&d_b,N*4));
    CHECK_RT(cudaMalloc(&d_fo,N*4)); CHECK_RT(cudaMalloc(&d_io,N*4));
    float*h=(float*)malloc(N*4);
    for(int i=0;i<N;i++)h[i]=(float)(i%1000)/1000.0f;
    CHECK_RT(cudaMemcpy(d_a,h,N*4,cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_b,h,N*4,cudaMemcpyHostToDevice));

    CHECK_CU(cuInit(0)); CUdevice dev; cuDeviceGet(&dev,0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx,dev); cuCtxSetCurrent(ctx);
    char dn[256]; cuDeviceGetName(dn,256,dev);
    printf("Device: %s\n",dn);
    printf("N = %d, %d iterations\n\n",N,ITERS);

    char*ptx=read_file(argv[1]);
    CUmodule mod; CHECK_CU(cuModuleLoadData(&mod,ptx));
    CUfunction oc_poly, oc_nbody, oc_mandel;
    CHECK_CU(cuModuleGetFunction(&oc_poly,mod,"bench_poly"));
    CHECK_CU(cuModuleGetFunction(&oc_nbody,mod,"bench_nbody"));
    CHECK_CU(cuModuleGetFunction(&oc_mandel,mod,"bench_mandelbrot"));

    CUdeviceptr ca=(CUdeviceptr)d_a, cb=(CUdeviceptr)d_b, cfo=(CUdeviceptr)d_fo, cio=(CUdeviceptr)d_io;

    printf("%-20s %10s %12s %8s\n","Kernel","nvcc(us)","OpenCUDA(us)","Ratio");
    printf("%-20s %10s %12s %8s\n","------","--------","-----------","-----");

    float tn,to;
    tn=time_ref(launch_ref_poly,d_fo,d_a,d_b,N,ITERS);
    to=time_cu(oc_poly,cfo,ca,cb,N,ITERS,0);
    printf("%-20s %10.1f %12.1f %7.2fx\n","poly_deg15",tn,to,to/tn);

    tn=time_ref(launch_ref_nbody,d_fo,d_a,d_b,N,ITERS);
    to=time_cu(oc_nbody,cfo,ca,cb,N,ITERS,0);
    printf("%-20s %10.1f %12.1f %7.2fx\n","nbody_64",tn,to,to/tn);

    tn=time_ref(launch_ref_mandel,d_io,d_a,d_b,N,ITERS);
    to=time_cu(oc_mandel,cio,ca,cb,N,ITERS,1);
    printf("%-20s %10.1f %12.1f %7.2fx\n","mandelbrot_256",tn,to,to/tn);

    printf("\nRatio = OpenCUDA/nvcc (lower is better, 1.0x = parity)\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_fo); cudaFree(d_io);
    free(h); free(ptx); cuModuleUnload(mod); cuDevicePrimaryCtxRelease(dev);
    return 0;
}
