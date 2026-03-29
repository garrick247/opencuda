/*
 * capture_alloc_submit.c — Inline-hook D3DKMTCreateAllocation and
 * D3DKMTSubmitCommand in gdi32.dll to capture NVIDIA's private driver
 * data formats for cuMemAlloc and cuLaunchKernel.
 *
 * Uses 14-byte absolute JMP (FF 25 00000000 + 8-byte addr) to hook the
 * function prologues directly — works regardless of when callers resolved
 * the pointer, including nvcuda64.dll's DllMain.
 *
 * Build:
 *   cl /W3 /O2 /nologo capture_alloc_submit.c /Fe:capture_alloc_submit.exe
 *
 * Output: alloc_capture.bin, submit_capture.bin in current directory.
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* -------------------------------------------------------------------------
 * D3DKMT types
 * ------------------------------------------------------------------------- */
typedef UINT D3DKMT_HANDLE;

typedef struct _D3DKMT_ALLOCATIONINFO {
    D3DKMT_HANDLE   hAllocation;        /* OUT */
    const void     *pSystemMem;
    void           *pPrivateDriverData;
    UINT            PrivateDriverDataSize;
    UINT            VidPnSourceId;
    union { UINT Value; } Flags;
} D3DKMT_ALLOCATIONINFO;

typedef struct _D3DKMT_CREATEALLOCATION {
    D3DKMT_HANDLE           hDevice;
    D3DKMT_HANDLE           hResource;
    D3DKMT_HANDLE           hGlobalShare;
    const void             *pPrivateRuntimeData;
    UINT                    PrivateRuntimeDataSize;
    const void             *pPrivateDriverData;     /* NV-specific resource descriptor */
    UINT                    PrivateDriverDataSize;
    UINT                    NumAllocations;
    D3DKMT_ALLOCATIONINFO  *pAllocationInfo;
    D3DKMT_HANDLE           hKMResource;            /* OUT */
    void                   *pPrivateRuntimeDataReturned;
    UINT                    PrivateRuntimeDataReturnedSize;
} D3DKMT_CREATEALLOCATION;

typedef struct _D3DKMT_SUBMITCOMMAND {
    UINT64          Commands;           /* GPU VA of command buffer */
    UINT            CommandLength;
    D3DKMT_HANDLE   hContext;
    UINT64          PresentHistoryToken;
    UINT            BroadcastContextCount;
    D3DKMT_HANDLE   BroadcastContext[64];
    void           *pPrivateDriverData;
    UINT            PrivateDriverDataSize;
    UINT            Flags;
    UINT64          MarkerLogType;
    UINT            RamdiskUsageInBytes;
    UINT            Pad;
} D3DKMT_SUBMITCOMMAND;

typedef LONG (WINAPI *PFN_D3DKMTCreateAllocation)(D3DKMT_CREATEALLOCATION *);
typedef LONG (WINAPI *PFN_D3DKMTSubmitCommand)(const D3DKMT_SUBMITCOMMAND *);

/* -------------------------------------------------------------------------
 * Inline hook: 14-byte absolute JMP
 *
 *   FF 25 00 00 00 00        JMP QWORD PTR [RIP+0]
 *   xx xx xx xx xx xx xx xx  target address
 * ------------------------------------------------------------------------- */
#define HOOK_SIZE 14

typedef struct {
    void                *target;
    uint8_t              saved[HOOK_SIZE];
    uint8_t              trampoline[HOOK_SIZE + 14]; /* saved bytes + jmp back */
    int                  installed;
} InlineHook;

static void
hook_install(InlineHook *h, void *target, void *hook_fn)
{
    h->target = target;

    /* Save original bytes */
    memcpy(h->saved, target, HOOK_SIZE);

    /* Build trampoline: execute saved bytes, then JMP back to target+HOOK_SIZE */
    memcpy(h->trampoline, h->saved, HOOK_SIZE);
    uint8_t *jmp = h->trampoline + HOOK_SIZE;
    jmp[0] = 0xFF; jmp[1] = 0x25;
    *(uint32_t *)(jmp + 2) = 0;
    *(uint64_t *)(jmp + 6) = (uint64_t)((uint8_t *)target + HOOK_SIZE);

    /* Make trampoline executable */
    DWORD old;
    VirtualProtect(h->trampoline, sizeof(h->trampoline), PAGE_EXECUTE_READWRITE, &old);

    /* Overwrite target prologue */
    VirtualProtect(target, HOOK_SIZE, PAGE_EXECUTE_READWRITE, &old);
    uint8_t patch[HOOK_SIZE];
    patch[0] = 0xFF; patch[1] = 0x25;
    *(uint32_t *)(patch + 2) = 0;
    *(uint64_t *)(patch + 6) = (uint64_t)hook_fn;
    memcpy(target, patch, HOOK_SIZE);
    VirtualProtect(target, HOOK_SIZE, old, &old);
    FlushInstructionCache(GetCurrentProcess(), target, HOOK_SIZE);

    h->installed = 1;
}

static void
hook_remove(InlineHook *h)
{
    if (!h->installed) return;
    DWORD old;
    VirtualProtect(h->target, HOOK_SIZE, PAGE_EXECUTE_READWRITE, &old);
    memcpy(h->target, h->saved, HOOK_SIZE);
    VirtualProtect(h->target, HOOK_SIZE, old, &old);
    FlushInstructionCache(GetCurrentProcess(), h->target, HOOK_SIZE);
    h->installed = 0;
}

/* -------------------------------------------------------------------------
 * Captured data storage
 * ------------------------------------------------------------------------- */
typedef struct {
    /* CreateAllocation */
    D3DKMT_HANDLE   hDevice;
    D3DKMT_HANDLE   hKMResource;
    uint64_t        GpuVirtualAddress;
    void           *pResourcePrivData;
    UINT            ResourcePrivDataSize;
    void           *pAllocPrivData;
    UINT            AllocPrivDataSize;
    UINT            NumAllocations;

    /* SubmitCommand */
    uint64_t        Commands;
    UINT            CommandLength;
    uint8_t        *CommandBuf;
    void           *pSubmitPrivData;
    UINT            SubmitPrivDataSize;
} CaptureData;

static CaptureData g_cap;
static InlineHook  g_hook_alloc;
static InlineHook  g_hook_submit;

/* -------------------------------------------------------------------------
 * Hooks
 * ------------------------------------------------------------------------- */
static LONG WINAPI
Hook_CreateAllocation(D3DKMT_CREATEALLOCATION *pData)
{
    /* Call original via trampoline */
    PFN_D3DKMTCreateAllocation orig =
        (PFN_D3DKMTCreateAllocation)g_hook_alloc.trampoline;
    LONG r = orig(pData);

    printf("[CreateAllocation] NTSTATUS=0x%08lX\n", r);
    printf("  hDevice        = 0x%08X\n", pData->hDevice);
    printf("  hKMResource    = 0x%08X  (OUT)\n", pData->hKMResource);
    printf("  NumAllocs      = %u\n", pData->NumAllocations);
    printf("  ResourcePriv   = %u bytes\n", pData->PrivateDriverDataSize);
    printf("  AllocPriv[0]   = %u bytes\n",
           pData->NumAllocations ? pData->pAllocationInfo[0].PrivateDriverDataSize : 0);
    if (pData->NumAllocations && pData->pAllocationInfo)
        printf("  hAllocation[0] = 0x%08X  (OUT)\n",
               pData->pAllocationInfo[0].hAllocation);

    /* Capture */
    g_cap.hDevice    = pData->hDevice;
    g_cap.hKMResource = pData->hKMResource;
    g_cap.NumAllocations = pData->NumAllocations;

    if (pData->pPrivateDriverData && pData->PrivateDriverDataSize) {
        g_cap.ResourcePrivDataSize = pData->PrivateDriverDataSize;
        g_cap.pResourcePrivData = malloc(pData->PrivateDriverDataSize);
        memcpy(g_cap.pResourcePrivData, pData->pPrivateDriverData,
               pData->PrivateDriverDataSize);
    }
    if (pData->NumAllocations && pData->pAllocationInfo &&
        pData->pAllocationInfo[0].pPrivateDriverData &&
        pData->pAllocationInfo[0].PrivateDriverDataSize) {
        g_cap.AllocPrivDataSize = pData->pAllocationInfo[0].PrivateDriverDataSize;
        g_cap.pAllocPrivData = malloc(g_cap.AllocPrivDataSize);
        memcpy(g_cap.pAllocPrivData,
               pData->pAllocationInfo[0].pPrivateDriverData,
               g_cap.AllocPrivDataSize);
    }
    return r;
}

static LONG WINAPI
Hook_SubmitCommand(const D3DKMT_SUBMITCOMMAND *pData)
{
    PFN_D3DKMTSubmitCommand orig =
        (PFN_D3DKMTSubmitCommand)g_hook_submit.trampoline;
    LONG r = orig(pData);

    printf("[SubmitCommand] NTSTATUS=0x%08lX\n", r);
    printf("  Commands (GPU VA) = 0x%016llX\n", (unsigned long long)pData->Commands);
    printf("  CommandLength     = %u bytes\n", pData->CommandLength);
    printf("  hContext          = 0x%08X\n", pData->hContext);
    printf("  PrivateDriverData = %u bytes\n", pData->PrivateDriverDataSize);

    /* Capture */
    g_cap.Commands = pData->Commands;
    g_cap.CommandLength = pData->CommandLength;
    if (pData->CommandLength) {
        /* Read command buffer from GPU VA via ReadProcessMemory — works if
         * it's a CPU-visible mapped region, which submission buffers are. */
        g_cap.CommandBuf = (uint8_t *)malloc(pData->CommandLength);
        SIZE_T read = 0;
        if (!ReadProcessMemory(GetCurrentProcess(),
                               (LPCVOID)pData->Commands,
                               g_cap.CommandBuf, pData->CommandLength, &read)) {
            printf("  (command buffer not CPU-readable: err=%lu)\n", GetLastError());
            free(g_cap.CommandBuf);
            g_cap.CommandBuf = NULL;
        } else {
            printf("  (command buffer read: %zu bytes)\n", read);
        }
    }
    if (pData->pPrivateDriverData && pData->PrivateDriverDataSize) {
        g_cap.SubmitPrivDataSize = pData->PrivateDriverDataSize;
        g_cap.pSubmitPrivData = malloc(pData->PrivateDriverDataSize);
        memcpy(g_cap.pSubmitPrivData, pData->pPrivateDriverData,
               pData->PrivateDriverDataSize);
    }
    return r;
}

/* -------------------------------------------------------------------------
 * Dump helpers
 * ------------------------------------------------------------------------- */
static void
dump_bin(const char *path, const void *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, 1, n, f); fclose(f); }
    printf("  -> %s  (%zu bytes)\n", path, n);
}

static void
dump_hex(const char *label, const void *data, size_t n)
{
    const uint8_t *p = (const uint8_t *)data;
    printf("%s (%zu bytes):\n", label, n);
    for (size_t i = 0; i < n && i < 256; i++) {
        if (i % 16 == 0) printf("  %04zx: ", i);
        printf("%02X ", p[i]);
        if (i % 16 == 15) printf("\n");
    }
    if (n % 16) printf("\n");
    if (n > 256) printf("  ... (%zu total)\n", n);
}

/* -------------------------------------------------------------------------
 * CUDA driver shim (same pattern as tracer)
 * ------------------------------------------------------------------------- */
typedef int CUresult;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
typedef void *CUmodule;
typedef void *CUfunction;

typedef CUresult (WINAPI *PFN_cuInit)(unsigned);
typedef CUresult (WINAPI *PFN_cuCtxCreate)(CUcontext *, unsigned, int);
typedef CUresult (WINAPI *PFN_cuCtxDestroy)(CUcontext);
typedef CUresult (WINAPI *PFN_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (WINAPI *PFN_cuMemFree)(CUdeviceptr);
typedef CUresult (WINAPI *PFN_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (WINAPI *PFN_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (WINAPI *PFN_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                               unsigned, unsigned, unsigned, unsigned,
                                               void *, void **, void **);

/* Minimal PTX no-op kernel, sm_89 (broad compat) */
static const char *PTX_NOOP =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    ".visible .entry noop(.param .u64 p) { ret; }\n";

/* -------------------------------------------------------------------------
 * main
 * ------------------------------------------------------------------------- */
int main(void)
{
    printf("capture_alloc_submit — inline-hook D3DKMTCreateAllocation + SubmitCommand\n\n");

    /* Install hooks BEFORE loading nvcuda */
    HMODULE gdi32 = LoadLibraryA("gdi32.dll");
    void *fn_alloc  = (void *)GetProcAddress(gdi32, "D3DKMTCreateAllocation");
    void *fn_submit = (void *)GetProcAddress(gdi32, "D3DKMTSubmitCommand");

    if (!fn_alloc || !fn_submit) {
        fprintf(stderr, "gdi32 exports not found\n");
        return 1;
    }
    printf("D3DKMTCreateAllocation @ %p\n", fn_alloc);
    printf("D3DKMTSubmitCommand    @ %p\n\n", fn_submit);

    hook_install(&g_hook_alloc,  fn_alloc,  Hook_CreateAllocation);
    hook_install(&g_hook_submit, fn_submit, Hook_SubmitCommand);
    printf("Hooks installed.\n\n");

    /* Now load CUDA and exercise the paths */
    HMODULE nvcuda = LoadLibraryA("nvcuda.dll");
    if (!nvcuda) { fprintf(stderr, "nvcuda.dll not found\n"); return 1; }

#define LOAD(T, fn) T fn = (T)GetProcAddress(nvcuda, #fn); \
    if (!fn) { fprintf(stderr, #fn " not found\n"); return 1; }
    LOAD(PFN_cuInit,             cuInit)
    LOAD(PFN_cuCtxCreate,        cuCtxCreate)
    LOAD(PFN_cuCtxDestroy,       cuCtxDestroy)
    LOAD(PFN_cuMemAlloc,         cuMemAlloc)
    LOAD(PFN_cuMemFree,          cuMemFree)
    LOAD(PFN_cuModuleLoadData,   cuModuleLoadData)
    LOAD(PFN_cuModuleGetFunction,cuModuleGetFunction)
    LOAD(PFN_cuLaunchKernel,     cuLaunchKernel)
#undef LOAD

    printf("=== cuInit ===\n");
    if (cuInit(0) != 0) { fprintf(stderr, "cuInit failed\n"); return 1; }

    printf("\n=== cuCtxCreate ===\n");
    CUcontext ctx = NULL;
    if (cuCtxCreate(&ctx, 0, 0) != 0) { fprintf(stderr, "cuCtxCreate failed\n"); return 1; }

    printf("\n=== cuMemAlloc (4 MB) ===\n");
    CUdeviceptr ptr = 0;
    if (cuMemAlloc(&ptr, 4 * 1024 * 1024) != 0) {
        fprintf(stderr, "cuMemAlloc failed\n"); return 1;
    }
    printf("  ptr = 0x%llx\n", (unsigned long long)ptr);

    printf("\n=== cuLaunchKernel ===\n");
    CUmodule mod = NULL;
    CUfunction fn = NULL;
    if (cuModuleLoadData(&mod, PTX_NOOP) != 0) {
        fprintf(stderr, "cuModuleLoadData failed\n"); return 1;
    }
    if (cuModuleGetFunction(&fn, mod, "noop") != 0) {
        fprintf(stderr, "cuModuleGetFunction failed\n"); return 1;
    }
    void *args[] = { &ptr };
    cuLaunchKernel(fn, 1,1,1, 1,1,1, 0, NULL, args, NULL);

    printf("\n=== cuMemFree ===\n");
    cuMemFree(ptr);

    cuCtxDestroy(ctx);

    /* Remove hooks before dumping */
    hook_remove(&g_hook_alloc);
    hook_remove(&g_hook_submit);

    /* Dump captured data */
    printf("\n=== Captured data ===\n");
    if (g_cap.pResourcePrivData) {
        dump_hex("CreateAllocation resource pPrivateDriverData",
                 g_cap.pResourcePrivData, g_cap.ResourcePrivDataSize);
        dump_bin("alloc_resource_priv.bin",
                 g_cap.pResourcePrivData, g_cap.ResourcePrivDataSize);
    }
    if (g_cap.pAllocPrivData) {
        dump_hex("CreateAllocation alloc[0] pPrivateDriverData",
                 g_cap.pAllocPrivData, g_cap.AllocPrivDataSize);
        dump_bin("alloc_alloc_priv.bin",
                 g_cap.pAllocPrivData, g_cap.AllocPrivDataSize);
    }
    if (g_cap.CommandBuf) {
        dump_hex("SubmitCommand command buffer", g_cap.CommandBuf, g_cap.CommandLength);
        dump_bin("submit_cmdbuf.bin", g_cap.CommandBuf, g_cap.CommandLength);
    }
    if (g_cap.pSubmitPrivData) {
        dump_hex("SubmitCommand pPrivateDriverData",
                 g_cap.pSubmitPrivData, g_cap.SubmitPrivDataSize);
        dump_bin("submit_priv.bin", g_cap.pSubmitPrivData, g_cap.SubmitPrivDataSize);
    }

    return 0;
}
