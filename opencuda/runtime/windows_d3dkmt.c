/*
 * windows_d3dkmt.c — OpenCUDA Windows runtime: D3DKMTEscape-based driver IPC.
 *
 * Implements cuInit / cuCtxCreate / cuCtxDestroy / cuMemAlloc / cuMemFree
 * by replaying the protocol captured from nvcuda64.dll via cuda_ioctl_tracer.
 *
 * ── IPC channel map (from session3 trace, 2026-03-23) ─────────────────────
 *
 *  Function      IPC path                    IOCTL/cmd          Notes
 *  ──────────    ──────────────────────────  ─────────────────  ───────────────
 *  cuInit        D3DKMTEscape (gdi32)        cmd=0x0500002B     KMT#1: 9308B static
 *                                            cmd=0x01000113     KMT#2:   69B static
 *  cuInit (DMA)  NtDeviceIoControlFile       IOCTL=0x08DE0004   DMA protect layer init
 *  cuCtxCreate   D3DKMTEscape                cmd=0x0200001E     KMT#3:   56B, driver writes ctx_handle at [0x20]
 *                                            cmd=0x0500002B     KMT#4:   69B static
 *  cuCtxDestroy  D3DKMTEscape                cmd=0x01000140     KMT#5-7: 54B x3 static
 *                                            cmd=0x0500002B     KMT#8:   69B, patch ctx_handle at [0x20]
 *  cuMemAlloc    D3DKMTCreateAllocation      GDI syscall        NOT via NtDeviceIoControlFile or Escape.
 *                                                               Goes through Win32k GDI syscall table
 *                                                               (NtGdiDdDDICreateAllocation or equivalent).
 *                                                               IAT hooks on gdi32.dll miss this if nvcuda64
 *                                                               calls the syscall stub directly (same as
 *                                                               NtDeviceIoControlFile bypass seen for RM path).
 *                                                               *** PRIVATE DRIVER DATA NOT YET CAPTURED ***
 *                                                               Needs tracer run with GDI syscall trampoline.
 *  cuMemFree     D3DKMTDestroyAllocation     GDI syscall        Same path as CreateAllocation.
 *  cuLaunchKernel D3DKMTSubmitCommand or    GDI syscall        NOT tested in session3 tracer. Likely same
 *                 push-buffer write                             GDI-syscall path. Needs capture.
 *
 * ── NVDA escape header (all D3DKMTEscape payloads) ────────────────────────
 *   [0x00] uint32 LE  magic1  = 0x4E564441  (wire bytes: 41 44 56 4E)
 *   [0x04] uint32 LE  version = 0x00010002
 *   [0x08] uint32 LE  size    (total payload size in bytes)
 *   [0x0C] uint32 LE  magic2  = 0x4E562A2A  (wire bytes: 2A 2A 56 4E)
 *   [0x10] uint32 LE  cmd     (RM command code)
 *
 * Build (MSVC, no special SDK beyond Windows headers):
 *   cl /W3 /O2 windows_d3dkmt.c /Fe:opencuda_runtime.exe
 */

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <initguid.h>

/* Minimal DXGI interface declarations — used only to enumerate adapters by vendor ID. */
typedef struct IDXGIFactory1 IDXGIFactory1;
typedef struct IDXGIAdapter1 IDXGIAdapter1;

typedef struct DXGI_ADAPTER_DESC1 {
    WCHAR  Description[128];
    UINT   VendorId;
    UINT   DeviceId;
    UINT   SubSysId;
    UINT   Revision;
    SIZE_T DedicatedVideoMemory;
    SIZE_T DedicatedSystemMemory;
    SIZE_T SharedSystemMemory;
    LUID   AdapterLuid;
    UINT   Flags;
} DXGI_ADAPTER_DESC1;

typedef struct IDXGIAdapter1Vtbl {
    /* IUnknown */
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIAdapter1 *, REFIID, void **);
    ULONG   (STDMETHODCALLTYPE *AddRef)        (IDXGIAdapter1 *);
    ULONG   (STDMETHODCALLTYPE *Release)       (IDXGIAdapter1 *);
    /* IDXGIObject */
    HRESULT (STDMETHODCALLTYPE *SetPrivateData)        (IDXGIAdapter1 *, REFGUID, UINT, const void *);
    HRESULT (STDMETHODCALLTYPE *SetPrivateDataInterface)(IDXGIAdapter1 *, REFGUID, const IUnknown *);
    HRESULT (STDMETHODCALLTYPE *GetPrivateData)        (IDXGIAdapter1 *, REFGUID, UINT *, void *);
    HRESULT (STDMETHODCALLTYPE *GetParent)             (IDXGIAdapter1 *, REFIID, void **);
    /* IDXGIAdapter */
    HRESULT (STDMETHODCALLTYPE *EnumOutputs)    (IDXGIAdapter1 *, UINT, void **);
    HRESULT (STDMETHODCALLTYPE *GetDesc)        (IDXGIAdapter1 *, void *);
    HRESULT (STDMETHODCALLTYPE *CheckInterfaceSupport)(IDXGIAdapter1 *, REFGUID, LARGE_INTEGER *);
    /* IDXGIAdapter1 */
    HRESULT (STDMETHODCALLTYPE *GetDesc1)(IDXGIAdapter1 *, DXGI_ADAPTER_DESC1 *);
} IDXGIAdapter1Vtbl;
struct IDXGIAdapter1 { IDXGIAdapter1Vtbl *lpVtbl; };

typedef struct IDXGIFactory1Vtbl {
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIFactory1 *, REFIID, void **);
    ULONG   (STDMETHODCALLTYPE *AddRef)        (IDXGIFactory1 *);
    ULONG   (STDMETHODCALLTYPE *Release)       (IDXGIFactory1 *);
    HRESULT (STDMETHODCALLTYPE *SetPrivateData)        (IDXGIFactory1 *, REFGUID, UINT, const void *);
    HRESULT (STDMETHODCALLTYPE *SetPrivateDataInterface)(IDXGIFactory1 *, REFGUID, const IUnknown *);
    HRESULT (STDMETHODCALLTYPE *GetPrivateData)        (IDXGIFactory1 *, REFGUID, UINT *, void *);
    HRESULT (STDMETHODCALLTYPE *GetParent)             (IDXGIFactory1 *, REFIID, void **);
    HRESULT (STDMETHODCALLTYPE *EnumAdapters)  (IDXGIFactory1 *, UINT, void **);
    HRESULT (STDMETHODCALLTYPE *MakeWindowAssociation)(IDXGIFactory1 *, HWND, UINT);
    HRESULT (STDMETHODCALLTYPE *GetWindowAssociation)(IDXGIFactory1 *, HWND *);
    HRESULT (STDMETHODCALLTYPE *CreateSwapChain)       (IDXGIFactory1 *, IUnknown *, void *, void **);
    HRESULT (STDMETHODCALLTYPE *CreateSoftwareAdapter) (IDXGIFactory1 *, HMODULE, void **);
    /* IDXGIFactory1 */
    HRESULT (STDMETHODCALLTYPE *EnumAdapters1) (IDXGIFactory1 *, UINT, IDXGIAdapter1 **);
    BOOL    (STDMETHODCALLTYPE *IsCurrent)     (IDXGIFactory1 *);
} IDXGIFactory1Vtbl;
struct IDXGIFactory1 { IDXGIFactory1Vtbl *lpVtbl; };

/* Returns the LUID of the first NVIDIA adapter (VendorId=0x10DE) via DXGI,
 * or {0,0} if not found.  Non-fatal if dxgi.dll is absent. */
static LUID
FindNvidiaLuidViaDxgi(void)
{
    LUID zero = {0, 0};
    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void **);
    HMODULE hDxgi = LoadLibraryA("dxgi.dll");
    if (!hDxgi) return zero;

    PFN_CreateDXGIFactory1 pfnCreate =
        (PFN_CreateDXGIFactory1)GetProcAddress(hDxgi, "CreateDXGIFactory1");
    if (!pfnCreate) { FreeLibrary(hDxgi); return zero; }

    /* IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387} */
    static const GUID IID_DXGIFactory1 = {
        0x770aae78, 0xf26f, 0x4dba,
        {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}
    };
    IDXGIFactory1 *factory = NULL;
    if (pfnCreate(&IID_DXGIFactory1, (void **)&factory) != 0 || !factory) {
        FreeLibrary(hDxgi);
        return zero;
    }

    LUID result = zero;
    IDXGIAdapter1 *adapter = NULL;
    for (UINT i = 0; factory->lpVtbl->EnumAdapters1(factory, i, &adapter) == 0; i++) {
        DXGI_ADAPTER_DESC1 desc = {0};
        adapter->lpVtbl->GetDesc1(adapter, &desc);
        fprintf(stderr, "DXGI[%u]: VendorId=0x%04X luid=%08lX:%08lX desc=%.64S\n",
                i, desc.VendorId,
                desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart,
                desc.Description);
        fflush(stderr);
        if (desc.VendorId == 0x10DE && result.LowPart == 0 && result.HighPart == 0) {
            result = desc.AdapterLuid;
        }
        adapter->lpVtbl->Release(adapter);
    }
    factory->lpVtbl->Release(factory);
    FreeLibrary(hDxgi);
    return result;
}

/* Full 9308-byte KMT#1 static blob (verified zero diffs across two capture sessions) */
#include "kmt1_blob.h"

/* 586-byte D3DKMTCreateAllocation private driver data blob (captured 2026-03-24).
 * Defines kKmtAlloc_Template[586] and patch-offset constants. */
#include "kmt_alloc_blob.h"

/* =========================================================================
 * D3DKMT types (subset — avoids dependency on dxgiddi.h / d3dkmthk.h)
 * ========================================================================= */

typedef UINT D3DKMT_HANDLE;

typedef enum _D3DKMT_ESCAPETYPE {
    D3DKMT_ESCAPE_DRIVERPRIVATE = 0,
} D3DKMT_ESCAPETYPE;

typedef struct _D3DKMT_ESCAPE {
    D3DKMT_HANDLE       hAdapter;              /* [0x00] */
    D3DKMT_HANDLE       hDevice;               /* [0x04] */
    D3DKMT_ESCAPETYPE   Type;                  /* [0x08] */
    UINT                Flags;                 /* [0x0C] — D3DDDI_ESCAPEFLAGS */
    VOID               *pPrivateDriverData;    /* [0x10] */
    UINT                PrivateDriverDataSize; /* [0x18] */
    D3DKMT_HANDLE       hContext;              /* [0x1C] */
} D3DKMT_ESCAPE;

typedef struct _D3DKMT_ADAPTERINFO {
    D3DKMT_HANDLE   hAdapter;
    LUID            AdapterLuid;
    ULONG           NumOfSources;
    BOOL            bPresentMoveRegionsPreferred;
} D3DKMT_ADAPTERINFO;

typedef struct _D3DKMT_ENUMADAPTERS2 {
    ULONG               NumAdapters;
    D3DKMT_ADAPTERINFO *pAdapters;
} D3DKMT_ENUMADAPTERS2;

typedef struct _D3DKMT_CREATEDEVICE {
    union {
        D3DKMT_HANDLE   hAdapter;
        ULONG_PTR       _pAdapter;  /* union with VOID* — makes field 8 bytes on 64-bit */
    };
    UINT            Flags;           /* [0x08] */
    D3DKMT_HANDLE   hDevice;         /* [0x0C] OUT */
    VOID           *pCommandBuffer;  /* [0x10] OUT */
    UINT            CommandBufferSize; /* [0x18] OUT */
} D3DKMT_CREATEDEVICE;

/* D3DKMTCreateAllocation / D3DKMTDestroyAllocation
 *
 * Used by cuMemAlloc / cuMemFree. The pPrivateDriverData in D3DDDI_ALLOCATIONINFO2
 * contains NVIDIA KMD-private allocation params (size, memory type, flags, etc.)
 * that have not yet been captured. A follow-up tracer run with a GDI-syscall
 * trampoline (targeting NtGdiDdDDICreateAllocation) is needed to recover them.
 *
 * Structure layout matches Windows DDK D3DKMT headers.
 */
typedef ULONG64 D3DGPU_VIRTUAL_ADDRESS;

typedef struct _D3DDDI_ALLOCATIONINFO2 {
    D3DKMT_HANDLE   hAllocation;        /* OUT: allocation handle */
    union {
        HANDLE      hSection;           /* system-mem backed */
        const VOID *pSystemMem;         /* NULL for VRAM */
    };
    const VOID     *pPrivateDriverData; /* NVIDIA KMD private data (IN) */
    UINT            PrivateDriverDataSize;
    UINT            VidPnSourceId;
    union {
        struct {
            UINT    Primary        : 1;
            UINT    Stereo         : 1;
            UINT    OverridePriority: 1;
            UINT    Reserved       :29;
        };
        UINT        Value;
    } Flags;
    D3DGPU_VIRTUAL_ADDRESS GpuVirtualAddress; /* OUT */
    UINT            Priority;
    UINT64          Reserved[5];
} D3DDDI_ALLOCATIONINFO2;

typedef struct _D3DKMT_CREATEALLOCATION {
    D3DKMT_HANDLE           hDevice;
    D3DKMT_HANDLE           hResource;
    D3DKMT_HANDLE           hGlobalShare;
    const VOID             *pPrivateRuntimeData;
    UINT                    PrivateRuntimeDataSize;
    const VOID             *pPrivateDriverData;
    UINT                    PrivateDriverDataSize;
    UINT                    NumAllocations;
    D3DDDI_ALLOCATIONINFO2 *pAllocationInfo2;
    D3DKMT_HANDLE           hAllocation;        /* OUT (single-alloc shortcut) */
    union {
        struct {
            UINT    ExistingSysMem     : 1;
            UINT    CreateShared       : 1;
            UINT    NtSecuritySharing  : 1;
            UINT    ReadOnly           : 1;
            UINT    CreateWriteCombined: 1;
            UINT    CreateCached       : 1;
            UINT    SwapChainBackBuffer: 1;
            UINT    Reserved           :25;
        };
        UINT        Value;
    } Flags;
    HANDLE              hPrivateRuntimeResourceHandle;
} D3DKMT_CREATEALLOCATION;

typedef struct _D3DKMT_DESTROYALLOCATION2 {
    D3DKMT_HANDLE   hDevice;
    D3DKMT_HANDLE   hResource;
    UINT            AllocationCount;
    const D3DKMT_HANDLE *phAllocationList;
    UINT            Flags;
} D3DKMT_DESTROYALLOCATION2;

typedef LONG (WINAPI *PFN_D3DKMTEscape)(const D3DKMT_ESCAPE *);
typedef LONG (WINAPI *PFN_D3DKMTEnumAdapters2)(D3DKMT_ENUMADAPTERS2 *);
typedef LONG (WINAPI *PFN_D3DKMTCreateDevice)(D3DKMT_CREATEDEVICE *);
typedef LONG (WINAPI *PFN_D3DKMTCreateAllocation)(D3DKMT_CREATEALLOCATION *);
typedef LONG (WINAPI *PFN_D3DKMTDestroyAllocation2)(const D3DKMT_DESTROYALLOCATION2 *);

typedef struct _D3DKMT_OPENADAPTERFROMLUID {
    LUID            AdapterLuid;   /* IN  — adapter LUID from DXGI */
    D3DKMT_HANDLE   hAdapter;      /* OUT — new kernel adapter handle */
} D3DKMT_OPENADAPTERFROMLUID;

typedef struct _D3DKMT_CLOSEADAPTER {
    D3DKMT_HANDLE   hAdapter;
} D3DKMT_CLOSEADAPTER;

typedef LONG (WINAPI *PFN_D3DKMTOpenAdapterFromLuid)(D3DKMT_OPENADAPTERFROMLUID *);
typedef LONG (WINAPI *PFN_D3DKMTCloseAdapter)(const D3DKMT_CLOSEADAPTER *);

/* =========================================================================
 * Static payload blobs (byte-exact captures from cuda_ioctl_tracer, 2026-03-23)
 *
 * Note on magic bytes: the NVDA and NV** fields are little-endian uint32 on wire,
 * so "NVDA" (0x4E564441) appears as bytes {0x41,0x44,0x56,0x4E}.
 * ========================================================================= */

/* KMT#2: cuInit version check — 69 bytes, cmd=0x01000113, fully static */
static const uint8_t kKmt2_VersionCheck[69] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x30,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x13,0x01,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x00,0xFF,0xFF,0xFF,
    0xFF,0x00,0x00,0x00,0x00
};

/* Pre-init: cmd=0x0100010E (52 bytes, Flags=0x8) — RM driver registration.
 * Sent to every adapter before KMT#1. Mirrors nvcuda64 ESCSYS#1,3,5 in trace.
 * Must use D3DKMT_ESCAPE.Flags=8 (NoAdapterSynchronization); hDevice=0. */
static const uint8_t kPreInit_DrvReg[52] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x34,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x0E,0x01,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x00,0x02,0x00,0x00
};

/* Pre-init: cmd=0x07000016 (56 bytes, Flags=0x8) — DMA layer registration.
 * Sent to every adapter after DrvReg. Mirrors nvdxgdmal64 ESCSYS#6 in trace. */
static const uint8_t kPreInit_DmaReg[56] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x38,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x16,0x00,0x00,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x00,0x02,0x00,0x00,0x00,0x02,0x00,0x00
};

/* KMT#3: cuCtxCreate alloc — 56 bytes, cmd=0x0200001E.
 * Send as mutable buffer; driver writes session RM handle in-place at [0x20..0x23].
 * [0x20..0x23] zeroed here — driver overwrites unconditionally. */
static const uint8_t kKmt3_CtxAllocTemplate[56] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x30,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x1E,0x00,0x00,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x32,0x00,0x00,0x00,0x05,0x00,0x00,0x00
};

/* KMT#4: cuCtxCreate params — 69 bytes, cmd=0x0500002B, fully static */
static const uint8_t kKmt4_CtxParams[69] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x45,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x2B,0x00,0x00,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x10,0x80,0x02,0xFF,0x10,0x00,0x02,0xFF,0x09,0x19,0x80,0x00,0x01,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x01
};

/* KMT#5: cuCtxDestroy cleanup — 54 bytes, cmd=0x01000140, fully static (sent x3) */
static const uint8_t kKmt5_CtxDestroyCleanup[54] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x30,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x40,0x01,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x01,0x00,0x00,0x00,0x00,0x00
};

/* KMT#8: cuCtxDestroy final — 69 bytes, cmd=0x0500002B.
 * Patch ctx handle from cuCtxCreate into [0x20..0x23] before sending. */
static const uint8_t kKmt8_CtxDestroyFinal[69] = {
    0x41,0x44,0x56,0x4E,0x02,0x00,0x01,0x00,0x45,0x00,0x00,0x00,0x2A,0x2A,0x56,0x4E,
    0x2B,0x00,0x00,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,  /* <-- [0x20..0x23]: patch ctx_handle here */
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,
    0x10,0x80,0x02,0xFF,0x10,0x00,0x02,0xFF,0x09,0x19,0x80,0x00,0x01,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00
};

/* =========================================================================
 * Runtime state
 * ========================================================================= */

typedef struct {
    HMODULE                     gdi32;
    PFN_D3DKMTEscape            Escape;
    PFN_D3DKMTEnumAdapters2     EnumAdapters2;
    PFN_D3DKMTCreateDevice      CreateDevice;
    PFN_D3DKMTCreateAllocation  CreateAllocation;   /* for opencuda_mem_alloc */
    PFN_D3DKMTDestroyAllocation2 DestroyAllocation2;/* for opencuda_mem_free */

    D3DKMT_HANDLE   hAdapter;
    D3DKMT_HANDLE   hDevice;

    uint32_t        ctx_handle;   /* written by KMT#3, consumed by KMT#8 */
    int             initialized;
} OpenCudaState;

static OpenCudaState g_state;

/* =========================================================================
 * Helpers
 * ========================================================================= */

static LONG
DoEscape(const void *data, UINT size)
{
    /* Always copy to a heap buffer — the NVIDIA RM writes back into the escape
     * buffer in-place.  Passing a static const .rodata pointer causes the write
     * to fault and the RM returns STATUS_INVALID_PARAMETER.
     *
     * For blobs where header.size < total array size: the header.size bytes are
     * the driver input; bytes beyond are the write-back area.  Pass header.size
     * as PrivateDriverDataSize so the driver sees the correct declared size. */
    const uint8_t *hdr = (const uint8_t *)data;
    UINT declared_size = (size >= 12)
        ? (UINT)(hdr[8] | ((UINT)hdr[9]<<8) | ((UINT)hdr[10]<<16) | ((UINT)hdr[11]<<24))
        : size;
    /* Use declared_size if it's <= total and non-zero, otherwise fall back to total. */
    UINT pass_size = (declared_size > 0 && declared_size <= size) ? declared_size : size;

    void *buf = malloc(size);  /* allocate full size so driver can write beyond declared_size */
    if (!buf) return (LONG)0xC0000017; /* STATUS_NO_MEMORY */
    memcpy(buf, data, size);

    D3DKMT_ESCAPE esc = {0};
    esc.hAdapter              = g_state.hAdapter;
    esc.hDevice               = g_state.hDevice;
    esc.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc.hContext              = 0;
    esc.pPrivateDriverData    = buf;
    esc.PrivateDriverDataSize = pass_size;
    esc.Flags                 = 0;
    fprintf(stderr, "DoEscape: hAdapter=0x%08X hDevice=0x%08X size=%u cmd=0x%08X\n",
            g_state.hAdapter, g_state.hDevice, pass_size,
            (size >= 20) ? (UINT)(((uint8_t*)buf)[16] | ((UINT)((uint8_t*)buf)[17]<<8) |
                                   ((UINT)((uint8_t*)buf)[18]<<16) | ((UINT)((uint8_t*)buf)[19]<<24)) : 0);
    fflush(stderr);
    LONG r = g_state.Escape(&esc);
    free(buf);
    return r;
}

/* Mutable variant — driver may write back into the buffer in-place. */
static LONG
DoEscapeMut(void *data, UINT size)
{
    D3DKMT_ESCAPE esc = {0};
    esc.hAdapter              = g_state.hAdapter;
    esc.hDevice               = g_state.hDevice;
    esc.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc.hContext              = 0;
    esc.pPrivateDriverData    = data;
    esc.PrivateDriverDataSize = size;
    esc.Flags                 = 0;
    return g_state.Escape(&esc);
}

/* =========================================================================
 * Adapter / device enumeration
 * ========================================================================= */

/*
 * PreInitNvAdminViaLuid() — open a SECOND kernel handle to the NVIDIA adapter
 * (via D3DKMTOpenAdapterFromLuid) and send DrvReg + DmaReg pre-init blobs to it.
 *
 * Trace analysis (ESCSYS#5/6 in cuda_ioctl_trace.txt):
 *   nvdxgdmal64 calls dxgdmalGetCallbacks(), which opens a separate D3DKMT handle
 *   to the NVIDIA adapter (0x40000580) — different from the handle used by nvcuda64
 *   for KMT#1/2 (0x40000480).  It sends DrvReg(52B)+DmaReg(56B) with Flags=0x8 to
 *   that second handle.  The NVIDIA RM appears to require this registration from the
 *   DMA layer client BEFORE it will accept KMT#2 on any handle for this adapter.
 *
 * g_state.gdi32 and g_state.Escape must be set before calling this.
 */
static void
PreInitNvAdminViaLuid(void)
{
    LUID nvLuid = FindNvidiaLuidViaDxgi();
    if (nvLuid.LowPart == 0 && nvLuid.HighPart == 0) {
        fprintf(stderr, "PreInitNvAdminViaLuid: no NVIDIA adapter from DXGI\n");
        fflush(stderr);
        return;
    }

    PFN_D3DKMTOpenAdapterFromLuid pfnOpen =
        (PFN_D3DKMTOpenAdapterFromLuid)GetProcAddress(
            g_state.gdi32, "D3DKMTOpenAdapterFromLuid");
    PFN_D3DKMTCloseAdapter pfnClose =
        (PFN_D3DKMTCloseAdapter)GetProcAddress(
            g_state.gdi32, "D3DKMTCloseAdapter");

    if (!pfnOpen || !pfnClose) {
        fprintf(stderr, "PreInitNvAdminViaLuid: D3DKMTOpenAdapterFromLuid or "
                        "D3DKMTCloseAdapter not found in gdi32\n");
        fflush(stderr);
        return;
    }

    D3DKMT_OPENADAPTERFROMLUID oa = {0};
    oa.AdapterLuid = nvLuid;
    LONG r = pfnOpen(&oa);
    fprintf(stderr, "PreInitNvAdminViaLuid: OpenFromLuid(NVIDIA luid=%08lX:%08lX) "
                    "-> 0x%08X  hAdapter=0x%08X\n",
            nvLuid.HighPart, nvLuid.LowPart, (UINT)r, oa.hAdapter);
    fflush(stderr);
    if (r != 0) return;

    uint8_t buf[56];   /* sized for the larger DmaReg blob */
    D3DKMT_ESCAPE pe = {0};
    pe.hAdapter = oa.hAdapter;
    pe.hDevice  = 0;
    pe.Type     = D3DKMT_ESCAPE_DRIVERPRIVATE;
    pe.Flags    = 8;   /* NoAdapterSynchronization — same as ESCSYS#5/6 */
    pe.hContext = 0;

    memcpy(buf, kPreInit_DrvReg, sizeof(kPreInit_DrvReg));
    pe.pPrivateDriverData    = buf;
    pe.PrivateDriverDataSize = sizeof(kPreInit_DrvReg);
    r = g_state.Escape(&pe);
    fprintf(stderr, "PreInitNvAdminViaLuid: DrvReg -> 0x%08X\n", (UINT)r);
    /* Dump DrvReg writeback to check if the RM assigns a client handle */
    fprintf(stderr, "PreInitNvAdminViaLuid: DrvReg buf:");
    for (int _i = 0; _i < (int)sizeof(kPreInit_DrvReg); _i++) {
        if (_i % 16 == 0) fprintf(stderr, "\n  [%02X]", _i);
        fprintf(stderr, " %02X", buf[_i]);
    }
    fprintf(stderr, "\n");

    memcpy(buf, kPreInit_DmaReg, sizeof(kPreInit_DmaReg));
    pe.pPrivateDriverData    = buf;
    pe.PrivateDriverDataSize = sizeof(kPreInit_DmaReg);
    r = g_state.Escape(&pe);
    fprintf(stderr, "PreInitNvAdminViaLuid: DmaReg -> 0x%08X\n", (UINT)r);
    fflush(stderr);

    D3DKMT_CLOSEADAPTER ca = { oa.hAdapter };
    pfnClose(&ca);
}

static int
FindNvidiaAdapter(void)
{
    /* Identify the NVIDIA adapter LUID via DXGI (VendorId = 0x10DE). */
    LUID nvLuid = FindNvidiaLuidViaDxgi();
    if (nvLuid.LowPart == 0 && nvLuid.HighPart == 0) {
        fprintf(stderr, "FindNvidiaAdapter: DXGI did not find NVIDIA adapter\n");
        return 0;
    }
    fprintf(stderr, "FindNvidiaAdapter: NVIDIA LUID=%08lX:%08lX\n",
            nvLuid.HighPart, nvLuid.LowPart);
    fflush(stderr);

    D3DKMT_ENUMADAPTERS2 ea = {0};
    /* First call with pAdapters=NULL: get count.
     * D3DKMTEnumAdapters2 returns STATUS_BUFFER_TOO_SMALL (0xC0000023)
     * on this probe call — that's expected, NumAdapters is still filled. */
    LONG rc = g_state.EnumAdapters2(&ea);
    if (rc != 0 && (ULONG)rc != 0xC0000023) {
        fprintf(stderr, "FindNvidiaAdapter: count query failed: NTSTATUS=0x%08X\n", (UINT)rc);
        return 0;
    }
    if (ea.NumAdapters == 0) {
        fprintf(stderr, "FindNvidiaAdapter: no adapters reported\n");
        return 0;
    }
    fprintf(stderr, "FindNvidiaAdapter: %u adapters\n", ea.NumAdapters);

    D3DKMT_ADAPTERINFO *adapters =
        (D3DKMT_ADAPTERINFO *)malloc(ea.NumAdapters * sizeof(*adapters));
    if (!adapters) return 0;
    ea.pAdapters = adapters;

    rc = g_state.EnumAdapters2(&ea);
    if (rc != 0) {
        fprintf(stderr, "FindNvidiaAdapter: fill query failed: NTSTATUS=0x%08X\n", (UINT)rc);
        free(adapters);
        return 0;
    }

    /*
     * Pre-init: send DrvReg(52B) to non-NVIDIA adapters only.
     *
     * Trace ESCSYS#1-4 show nvcuda64 sends DrvReg+568B to adapters 0x40000200
     * and 0x400002C0 (Intel/MSFT handles) — but NEVER to the NVIDIA display
     * adapter (0x40000480).  The NVIDIA adapter's pre-init is handled separately
     * via PreInitNvAdminViaLuid() (ESCSYS#5/6, a second LUID-opened handle).
     * Sending DrvReg to the same handle used for KMT#1/2 appears to be incorrect.
     *
     * We skip the NVIDIA adapter here (luidMatch) and only attempt non-NVIDIA adapters.
     * Non-NVIDIA adapters may return non-zero NTSTATUS for DrvReg — that's expected
     * (Intel/MSFT don't implement this command), so we ignore errors.
     */
    for (ULONG i = 0; i < ea.NumAdapters; i++) {
        D3DKMT_HANDLE ha     = adapters[i].hAdapter;
        LUID          adLuid = adapters[i].AdapterLuid;
        int isNvidia = (adLuid.LowPart  == nvLuid.LowPart &&
                        adLuid.HighPart == nvLuid.HighPart);
        if (isNvidia) {
            fprintf(stderr, "PreInit: skipping NVIDIA display adapter [%lu] ha=0x%08X "
                            "(pre-init handled by PreInitNvAdminViaLuid)\n", i, ha);
            continue;
        }
        uint8_t buf[52];
        D3DKMT_ESCAPE pe = {0};
        pe.hAdapter              = ha;
        pe.hDevice               = 0;
        pe.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
        pe.Flags                 = 8;  /* NoAdapterSynchronization */
        pe.hContext              = 0;
        memcpy(buf, kPreInit_DrvReg, sizeof(kPreInit_DrvReg));
        pe.pPrivateDriverData    = buf;
        pe.PrivateDriverDataSize = sizeof(kPreInit_DrvReg);
        LONG r = g_state.Escape(&pe);
        fprintf(stderr, "PreInit DrvReg [%lu] ha=0x%08X -> 0x%08X%s\n",
                i, ha, (UINT)r, r == 0 ? "" : " (ignored)");
    }
    fflush(stderr);

    /*
     * Find the adapter whose LUID matches the NVIDIA LUID from DXGI.
     * Then create a KMT device on it and send KMT#1 (RM init).
     */
    int found = 0;
    for (ULONG i = 0; i < ea.NumAdapters && !found; i++) {
        D3DKMT_HANDLE ha      = adapters[i].hAdapter;
        LUID          adLuid  = adapters[i].AdapterLuid;

        int luidMatch = (adLuid.LowPart  == nvLuid.LowPart &&
                         adLuid.HighPart == nvLuid.HighPart);

        fprintf(stderr, "FindNvidiaAdapter: [%lu] hAdapter=0x%08X src=%lu luid=%08lX:%08lX %s\n",
                i, ha, adapters[i].NumOfSources,
                adLuid.HighPart, adLuid.LowPart,
                luidMatch ? "<-- NVIDIA" : "");
        fflush(stderr);

        if (!luidMatch) continue;

        /* LUID matched — create device and send KMT#1 */
        D3DKMT_CREATEDEVICE cd = {0};
        cd.hAdapter = ha;
        cd.Flags    = 0;  /* Flags=0 confirmed from trace — nvdxgdmal64.dll uses 0 */
        rc = g_state.CreateDevice(&cd);
        if (rc != 0) {
            fprintf(stderr, "FindNvidiaAdapter: [%lu]   CreateDevice failed: 0x%08X\n", i, (UINT)rc);
            fflush(stderr);
            break;
        }

        D3DKMT_HANDLE hd = cd.hDevice;
        fprintf(stderr, "FindNvidiaAdapter: [%lu]   hDevice=0x%08X\n", i, hd);
        fflush(stderr);

        /* Diagnostic: probe KMT#2 BEFORE KMT#1 to check if KMT#1 causes the failure.
         * If this also returns 0xC000000D, KMT#1 is not the problem. */
        {
            uint8_t _b2[sizeof(kKmt2_VersionCheck)];
            memcpy(_b2, kKmt2_VersionCheck, sizeof(_b2));
            D3DKMT_ESCAPE _e2 = {0};
            _e2.hAdapter = ha; _e2.hDevice = hd;
            _e2.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
            _e2.pPrivateDriverData = _b2;
            _e2.PrivateDriverDataSize = sizeof(_b2);
            LONG _r2 = g_state.Escape(&_e2);
            fprintf(stderr, "FindNvidiaAdapter: KMT#2 BEFORE KMT#1: NTSTATUS=0x%08X\n", (UINT)_r2);
            fflush(stderr);
        }

        /* Send KMT#1 (RM init, 9308B) on the confirmed NVIDIA adapter.
         * Use a mutable heap copy — the RM may write back into the buffer in-place
         * (same pattern as KMT#3).  A static const buffer would cause the write to
         * fault and return STATUS_INVALID_PARAMETER. */
        uint8_t *buf1 = (uint8_t *)malloc(sizeof(kKmt1_RmInit));
        if (!buf1) break;
        memcpy(buf1, kKmt1_RmInit, sizeof(kKmt1_RmInit));
        /* Send verbatim — kmt1_blob.h is the post-call capture and matches the
         * driver's expected input layout, including byte 44 = 0x01.  Do not zero
         * anything; bytes 32-35 are the driver's writeback slot (already 0 in the
         * pre-call image) and bytes 44-47 (0x01000000) are RM config inputs. */

        D3DKMT_ESCAPE esc = {0};
        esc.hAdapter              = ha;
        esc.hDevice               = hd;
        esc.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
        esc.pPrivateDriverData    = buf1;
        esc.PrivateDriverDataSize = sizeof(kKmt1_RmInit);
        rc = g_state.Escape(&esc);

        fprintf(stderr, "FindNvidiaAdapter: KMT1=%s(0x%08X)\n",
                rc == 0 ? "OK" : "fail", (UINT)rc);
        if (rc == 0) {
            /* Dump first 96 bytes the driver wrote back — looking for RM client handle.
             * The original template has: bytes 0-19 = NVDA header (should be unchanged),
             * bytes 20+ = payload (look for non-zero writes from the RM). */
            fprintf(stderr, "FindNvidiaAdapter: KMT1 writeback (first 96 bytes):\n");
            for (int di = 0; di < 96; di += 16) {
                fprintf(stderr, "  [%02X]", di);
                for (int j = di; j < di+16 && j < 96; j++)
                    fprintf(stderr, " %02X", buf1[j]);
                fprintf(stderr, "\n");
            }

            /* Also check for any non-zero writeback beyond byte 20 */
            int first_change = -1;
            for (int di = 20; di < (int)sizeof(kKmt1_RmInit); di++) {
                if (buf1[di] != kKmt1_RmInit[di]) {
                    first_change = di;
                    break;
                }
            }
            if (first_change >= 0)
                fprintf(stderr, "FindNvidiaAdapter: first writeback diff at offset 0x%X = 0x%02X\n",
                        first_change, buf1[first_change]);
            else
                fprintf(stderr, "FindNvidiaAdapter: no writeback diffs detected in payload\n");
        }
        fflush(stderr);
        free(buf1);

        if (rc == 0) {
            g_state.hAdapter = ha;
            g_state.hDevice  = hd;
            found = 1;
        }
    }

    free(adapters);
    return found;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

/*
 * opencuda_init() — equivalent to cuInit(0).
 *
 * Sequence: DMA layer init (NvAdminDevice IOCTL 0x08DE0004)
 *           -> enumerate adapters, probe KMT#2 per adapter to find NVIDIA one
 *           -> KMT#1 (RM init, 9308B) -> KMT#2 (version check, 69B).
 * Returns 0 on success, negative on error.
 */
int
opencuda_init(void)
{
    if (g_state.initialized) return 0;

    g_state.gdi32 = LoadLibraryA("gdi32.dll");
    if (!g_state.gdi32) { fprintf(stderr, "opencuda_init: gdi32.dll load failed\n"); return -1; }

    g_state.Escape = (PFN_D3DKMTEscape)GetProcAddress(g_state.gdi32, "D3DKMTEscape");
    if (!g_state.Escape) { fprintf(stderr, "opencuda_init: D3DKMTEscape not found\n"); return -1; }
    g_state.EnumAdapters2 = (PFN_D3DKMTEnumAdapters2)GetProcAddress(g_state.gdi32, "D3DKMTEnumAdapters2");
    if (!g_state.EnumAdapters2) { fprintf(stderr, "opencuda_init: D3DKMTEnumAdapters2 not found\n"); return -1; }
    g_state.CreateDevice = (PFN_D3DKMTCreateDevice)GetProcAddress(g_state.gdi32, "D3DKMTCreateDevice");
    if (!g_state.CreateDevice) { fprintf(stderr, "opencuda_init: D3DKMTCreateDevice not found\n"); return -1; }

    /* Optional — needed for opencuda_mem_alloc; non-fatal if absent */
    g_state.CreateAllocation   = (PFN_D3DKMTCreateAllocation)GetProcAddress(g_state.gdi32, "D3DKMTCreateAllocation");
    g_state.DestroyAllocation2 = (PFN_D3DKMTDestroyAllocation2)GetProcAddress(g_state.gdi32, "D3DKMTDestroyAllocation2");

    /* DMA protection layer init — must happen before any D3DKMTEscape RM call.
     * nvcuda64 opens nvdxgdmal64.dll via DriverStore path; from a normal process
     * the same device is reachable via the NvAdminDevice symlink. */
    {
        HANDLE hDma = CreateFileW(L"\\\\.\\NvAdminDevice",
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL, OPEN_EXISTING, 0, NULL);
        if (hDma == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "opencuda_init: CreateFileW NvAdminDevice failed: err=%lu\n", GetLastError());
            return -1;
        }

        uint8_t dma_in[136]  = {0};
        uint8_t dma_out[136] = {0};
        dma_in[0] = 0x02;  /* first dword = 0x00000002 LE */
        DWORD returned = 0;
        BOOL ok = DeviceIoControl(hDma, 0x08DE0004,
                                  dma_in, sizeof(dma_in),
                                  dma_out, sizeof(dma_out),
                                  &returned, NULL);
        DWORD dma_err = GetLastError();
        CloseHandle(hDma);

        if (!ok && returned == 0) {
            fprintf(stderr, "opencuda_init: DMA layer IOCTL failed ok=%d err=%lu\n", ok, dma_err);
            return -1;
        }
        fprintf(stdout, "DMA layer init: ok=%d returned=%lu\n", ok, returned);
    }

    /* Try NvAPI_Initialize() first.
     * In the trace, nvapi_QueryInterface/nvapi_Direct_GetMethod are called by nvcuda64
     * BEFORE ESCSYS#1.  nvapi64 may call D3DKMTEscape via its own syscall stub
     * (bypassing the tracer's hook on win32u.dll), setting RM state that KMT#2 needs. */
    {
        HMODULE hNvapi = LoadLibraryA("nvapi64.dll");
        if (hNvapi) {
            /* nvapi_QueryInterface(0x0150E828) → NvAPI_Initialize */
            typedef void* (*PFN_nvapi_QueryInterface)(UINT);
            PFN_nvapi_QueryInterface pfnQI =
                (PFN_nvapi_QueryInterface)GetProcAddress(hNvapi, "nvapi_QueryInterface");
            if (pfnQI) {
                typedef int (CDECL *PFN_NvAPI_Initialize)(void);
                PFN_NvAPI_Initialize pfnInit =
                    (PFN_NvAPI_Initialize)pfnQI(0x0150E828U);
                if (pfnInit) {
                    int nr = pfnInit();
                    fprintf(stderr, "NvAPI_Initialize -> %d\n", nr);
                } else {
                    fprintf(stderr, "NvAPI_Initialize not found via QueryInterface\n");
                }
            } else {
                fprintf(stderr, "nvapi64.dll: nvapi_QueryInterface not found\n");
            }
            fflush(stderr);
            /* NOTE: intentionally NOT calling FreeLibrary — keep nvapi64 loaded
             * so any background threads it started stay alive during our KMT#2 test. */
        } else {
            fprintf(stderr, "nvapi64.dll load failed: err=%lu\n", GetLastError());
        }
    }

    /* Pre-init to nvdxgdmal's second NVIDIA adapter handle — mirrors ESCSYS#5/6.
     * nvdxgdmal64 (via dxgdmalGetCallbacks) opens a SEPARATE D3DKMT handle to the
     * NVIDIA adapter (not the one used for KMT#1/2) and sends DrvReg+DmaReg to it.
     * The NVIDIA RM requires this DMA-layer registration before KMT#2 succeeds.
     * We replicate this by opening a second handle via D3DKMTOpenAdapterFromLuid. */
    PreInitNvAdminViaLuid();

    /* FindNvidiaAdapter enumerates adapters and sends KMT#1 (RM init) as probe.
     * On success g_state.hAdapter/hDevice are set and KMT#1 has already been sent. */
    if (!FindNvidiaAdapter()) { fprintf(stderr, "opencuda_init: FindNvidiaAdapter failed\n"); return -1; }

    /* KMT#2: version check (69B) — probe both full size (69) and declared size (48) */
    {
        static const UINT try_sizes[] = {sizeof(kKmt2_VersionCheck), 48};
        int kmt2_ok = 0;
        for (int si = 0; si < 2 && !kmt2_ok; si++) {
            uint8_t *buf2 = (uint8_t *)malloc(sizeof(kKmt2_VersionCheck));
            if (!buf2) break;
            memcpy(buf2, kKmt2_VersionCheck, sizeof(kKmt2_VersionCheck));
            D3DKMT_ESCAPE esc2 = {0};
            esc2.hAdapter              = g_state.hAdapter;
            esc2.hDevice               = g_state.hDevice;
            esc2.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
            esc2.hContext              = 0;
            esc2.pPrivateDriverData    = buf2;
            esc2.PrivateDriverDataSize = try_sizes[si];
            esc2.Flags                 = 0;
            LONG r2 = g_state.Escape(&esc2);
            fprintf(stderr, "opencuda_init: KMT#2 size=%u: NTSTATUS=0x%08X%s\n",
                    try_sizes[si], (UINT)r2, r2==0?" OK":"");
            fflush(stderr);
            if (r2 == 0) kmt2_ok = 1;
            free(buf2);
        }
    }

    g_state.initialized = 1;
    return 0;
}

/*
 * opencuda_ctx_create() — equivalent to cuCtxCreate.
 *
 * Sends KMT#3 as a mutable buffer; driver writes the RM context handle
 * into [0x20..0x23] in-place. Stores handle for destroy. Then sends KMT#4.
 * Returns 0 on success.
 */
int
opencuda_ctx_create(void)
{
    /* KMT#3: alloc — mutable so driver can write handle back */
    uint8_t buf3[sizeof(kKmt3_CtxAllocTemplate)];
    memcpy(buf3, kKmt3_CtxAllocTemplate, sizeof(buf3));

    LONG r3 = DoEscapeMut(buf3, sizeof(buf3));
    if (r3 != 0) {
        fprintf(stderr, "opencuda_ctx_create: KMT#3 failed: NTSTATUS=0x%08X\n", (UINT)r3);
        return -1;
    }

    /* Read back the 4-byte RM handle the driver placed at [0x20..0x23].
     * All 4 bytes are session-derived (e.g. 0xB0DE04B8, 0x9220B6B8 across sessions). */
    memcpy(&g_state.ctx_handle, &buf3[0x20], 4);

    /* KMT#4: ctx params (69B, fully static) */
    LONG r4 = DoEscape(kKmt4_CtxParams, sizeof(kKmt4_CtxParams));
    if (r4 != 0) return -1;

    return 0;
}

/*
 * opencuda_ctx_destroy() — equivalent to cuCtxDestroy.
 *
 * Sends KMT#5 x3 (static cleanup), then KMT#8 with ctx handle patched in.
 * Returns 0 on success.
 */
int
opencuda_ctx_destroy(void)
{
    for (int i = 0; i < 3; i++) {
        LONG r = DoEscape(kKmt5_CtxDestroyCleanup, sizeof(kKmt5_CtxDestroyCleanup));
        if (r != 0) return -1;
    }

    uint8_t buf8[sizeof(kKmt8_CtxDestroyFinal)];
    memcpy(buf8, kKmt8_CtxDestroyFinal, sizeof(buf8));
    memcpy(&buf8[0x20], &g_state.ctx_handle, 4);

    LONG r8 = DoEscape(buf8, sizeof(buf8));
    if (r8 != 0) return -1;

    g_state.ctx_handle = 0;
    return 0;
}

/* forward declaration */
int opencuda_mem_free(D3DKMT_HANDLE *handles, UINT nhandles);

/* =========================================================================
 * Memory allocation
 *
 * cuMemAlloc routes through D3DKMTCreateAllocation (GDI syscall path), NOT
 * D3DKMTEscape. The NVIDIA KMD private driver data format for a VRAM alloc
 * is not yet captured — pPrivateDriverData and PrivateDriverDataSize below
 * are placeholders. A follow-up capture run is needed:
 *
 *   Capture target: hook NtGdiDdDDICreateAllocation (Win32k syscall) or
 *                   D3DKMTCreateAllocation in gdi32.dll via byte-patching
 *                   (not IAT — nvcuda64 likely calls the GDI syscall stub
 *                   directly, same pattern as the NtDeviceIoControlFile bypass).
 *
 *   Once captured: paste pPrivateDriverData blob for a VRAM allocation into
 *   kKmtAlloc_PrivateData[] below and set kKmtAlloc_PrivateDataSize.
 * ========================================================================= */

/*
 * opencuda_mem_alloc() — allocate GPU VRAM via D3DKMTCreateAllocation.
 *
 * Uses the captured 586-byte NVIDIA KMD private driver data blob.
 * Splits allocations > 64MB into multiple calls (KALLOC_MAX_CHUNK per call),
 * mirroring the nvcuda64 behaviour confirmed in session4 trace.
 *
 * OUT fields after call:
 *   out_handles[0..n-1]  — kernel allocation handles (one per chunk)
 *   *out_nhandles        — number of handles
 *   gpuVA                — NOT filled; requires a subsequent
 *                          NtGdiDdDDIMapGpuVirtualAddress call (not yet captured).
 *
 * Returns 0 on success, negative on error.
 */
int
opencuda_mem_alloc(uint64_t size_bytes,
                   D3DKMT_HANDLE *out_handles, UINT *out_nhandles)
{
    if (!out_handles || !out_nhandles) return -1;
    *out_nhandles = 0;

    if (!g_state.CreateAllocation) return -2;

    uint64_t remaining = size_bytes;
    UINT     n         = 0;

    while (remaining > 0) {
        uint32_t chunk = (remaining > KALLOC_MAX_CHUNK)
                         ? KALLOC_MAX_CHUNK
                         : (uint32_t)remaining;

        /* Mutable copy of the template — patch the three size fields */
        uint8_t blob[KALLOC_BLOB_SIZE];
        memcpy(blob, kKmtAlloc_Template, KALLOC_BLOB_SIZE);
        memcpy(&blob[KALLOC_OFF_SIZE],    &chunk, 4);
        uint32_t chunk_m1 = chunk - 1;
        memcpy(&blob[KALLOC_OFF_SIZE_M1], &chunk_m1, 4);
        memcpy(&blob[KALLOC_OFF_SIZE2],   &chunk, 4);

        D3DDDI_ALLOCATIONINFO2 ai = {0};
        ai.pSystemMem            = NULL;
        ai.pPrivateDriverData    = blob;
        ai.PrivateDriverDataSize = KALLOC_BLOB_SIZE;

        /* D3DKMT_CREATEALLOCATION with the Win11 26100 extended layout.
         * NumAllocations at [0x2C], pAllocationInfo2 at [0x30]. */
        struct {
            D3DKMT_HANDLE           hDevice;
            D3DKMT_HANDLE           hResource;
            D3DKMT_HANDLE           hGlobalShare;
            UINT                    _pad0;
            const VOID             *pPrivateRuntimeData;
            UINT                    PrivateRuntimeDataSize;
            UINT                    _ext1;
            const VOID             *_ext_ptr;
            UINT                    _ext2;
            UINT                    NumAllocations;    /* [0x2C] */
            D3DDDI_ALLOCATIONINFO2 *pAllocationInfo2;  /* [0x30] */
        } ca;
        memset(&ca, 0, sizeof(ca));
        ca.hDevice           = g_state.hDevice;
        ca.NumAllocations    = 1;
        ca.pAllocationInfo2  = &ai;

        LONG r = g_state.CreateAllocation((D3DKMT_CREATEALLOCATION *)&ca);
        if (r != 0) {
            /* Free any already-allocated chunks on partial failure */
            for (UINT i = 0; i < n; i++) opencuda_mem_free(&out_handles[i], 1);
            *out_nhandles = 0;
            return -3;
        }

        out_handles[n++] = ai.hAllocation;
        remaining -= chunk;
    }

    *out_nhandles = n;
    return 0;
    /* TODO: call NtGdiDdDDIMapGpuVirtualAddress to obtain the GPU VA.
     * The driver fills ai.GpuVirtualAddress = 0 currently — VA mapping
     * needs a separate hook run on NtGdiDdDDIMapGpuVirtualAddress. */
}

/*
 * opencuda_mem_free() — free chunk handles from opencuda_mem_alloc.
 */
int
opencuda_mem_free(D3DKMT_HANDLE *handles, UINT nhandles)
{
    if (!g_state.DestroyAllocation2 || !handles || !nhandles) return -1;

    D3DKMT_DESTROYALLOCATION2 da = {0};
    da.hDevice          = g_state.hDevice;
    da.AllocationCount  = nhandles;
    da.phAllocationList = handles;

    LONG r = g_state.DestroyAllocation2(&da);
    return (r == 0) ? 0 : -1;
}

/* =========================================================================
 * IAT hook — intercept nvcuda64.dll's calls to D3DKMTEscape
 *
 * Strategy: after LoadLibraryA("nvcuda.dll"), scan nvcuda64's Import Address
 * Table for entries that currently point to the real D3DKMTEscape.  Replace
 * them with our hook function pointer.  The hook logs every escape — in
 * particular dumps the full payload when cmd=0x01000113 (KMT#2) — then
 * forwards the call to g_state.Escape (the real function, captured at
 * startup via GetProcAddress before nvcuda was ever loaded).
 *
 * IAT patching avoids the syscall-stub trampoline problem entirely: we only
 * write a pointer in a data section, not bytes inside the function body.
 * ========================================================================= */

static UINT s_hook_call_idx = 0;
static void **s_iat_entry   = NULL;   /* nvcuda IAT slot we patched */
static void  *s_iat_orig    = NULL;   /* original value (real D3DKMTEscape) */

/* Hook — called instead of D3DKMTEscape from nvcuda64 code paths. */
static LONG WINAPI
EscapeHookFn(const D3DKMT_ESCAPE *esc)
{
    UINT idx = ++s_hook_call_idx;
    UINT cmd = 0;
    if (esc && esc->pPrivateDriverData && esc->PrivateDriverDataSize >= 20) {
        const uint8_t *p = (const uint8_t *)esc->pPrivateDriverData;
        cmd = p[16] | ((UINT)p[17]<<8) | ((UINT)p[18]<<16) | ((UINT)p[19]<<24);
    }
    fprintf(stderr, "[H#%03u] ha=0x%08X hd=0x%08X fl=0x%X sz=%u cmd=0x%08X\n",
            idx,
            esc ? esc->hAdapter : 0, esc ? esc->hDevice : 0,
            esc ? esc->Flags : 0, esc ? esc->PrivateDriverDataSize : 0,
            cmd);
    /* Dump full payload for KMT#2 so we can compare against kKmt2_VersionCheck */
    if (cmd == 0x01000113 && esc && esc->pPrivateDriverData) {
        UINT sz = esc->PrivateDriverDataSize;
        const uint8_t *p = (const uint8_t *)esc->pPrivateDriverData;
        fprintf(stderr, "[H] *** KMT2 PAYLOAD size=%u ***\n", sz);
        for (UINT ri = 0; ri < sz && ri < 256; ri += 16) {
            fprintf(stderr, "  [%02X]", ri);
            for (UINT rj = ri; rj < ri+16 && rj < sz; rj++)
                fprintf(stderr, " %02X", p[rj]);
            fprintf(stderr, "\n");
        }
    }
    fflush(stderr);

    /* Forward to the real D3DKMTEscape via g_state.Escape.
     * g_state.Escape was captured before nvcuda loaded and doesn't go through
     * nvcuda's (now-patched) IAT, so there's no recursion. */
    LONG r = g_state.Escape(esc);
    if (cmd)
        fprintf(stderr, "[H#%03u] -> 0x%08X\n", idx, (UINT)r);
    fflush(stderr);
    return r;
}

/* Scan hMod's IAT for entries matching target_fn; replace with hook_fn.
 * Returns the patched entry's address (NULL if not found). */
static void **
PatchIatEntry(HMODULE hMod, void *target_fn, void *hook_fn)
{
    uint8_t *base = (uint8_t *)hMod;
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
    if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE) return NULL;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return NULL;

    DWORD imp_rva = nt->OptionalHeader
                       .DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
                       .VirtualAddress;
    if (!imp_rva) return NULL;

    IMAGE_IMPORT_DESCRIPTOR *imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + imp_rva);
    for (; imp->Name; imp++) {
        IMAGE_THUNK_DATA64 *iat = (IMAGE_THUNK_DATA64 *)(base + imp->FirstThunk);
        for (UINT i = 0; iat[i].u1.Function; i++) {
            void **slot = (void **)&iat[i].u1.Function;
            if (*slot != target_fn) continue;

            fprintf(stderr, "IATHook: found %p in %s[%u] -> patching\n",
                    target_fn, (const char *)(base + imp->Name), i);
            DWORD old_prot, dummy;
            VirtualProtect(slot, sizeof(void *), PAGE_READWRITE, &old_prot);
            *slot = hook_fn;
            VirtualProtect(slot, sizeof(void *), old_prot, &dummy);
            return slot;
        }
    }
    return NULL;
}

/* Dump all imports of hMod to stderr for diagnostic purposes. */
static void
DumpImports(HMODULE hMod, const char *mod_label)
{
    uint8_t *base = (uint8_t *)hMod;
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
    if (!dos || dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    DWORD imp_rva = nt->OptionalHeader
                       .DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
                       .VirtualAddress;
    if (!imp_rva) { fprintf(stderr, "DumpImports(%s): no import table\n", mod_label); return; }
    IMAGE_IMPORT_DESCRIPTOR *imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + imp_rva);
    fprintf(stderr, "DumpImports(%s): import table:\n", mod_label);
    for (; imp->Name; imp++) {
        const char *dll = (const char *)(base + imp->Name);
        /* Only show DLLs with "gdi" or "kernel" or "d3d" in the name */
        if (_stricmp(dll, "gdi32.dll") == 0 || _stricmp(dll, "gdi32full.dll") == 0 ||
            _stricmp(dll, "api-ms-win-gdi-private-l1-1-0.dll") == 0) {
            IMAGE_THUNK_DATA64 *orig = imp->OriginalFirstThunk ?
                (IMAGE_THUNK_DATA64 *)(base + imp->OriginalFirstThunk) : NULL;
            IMAGE_THUNK_DATA64 *iat  = (IMAGE_THUNK_DATA64 *)(base + imp->FirstThunk);
            fprintf(stderr, "  DLL: %s\n", dll);
            for (UINT i = 0; iat[i].u1.Function; i++) {
                const char *fn_name = "?";
                if (orig && !IMAGE_SNAP_BY_ORDINAL64(orig[i].u1.Ordinal))
                    fn_name = (const char *)((IMAGE_IMPORT_BY_NAME *)(base + (DWORD)orig[i].u1.AddressOfData))->Name;
                fprintf(stderr, "    [%u] %-40s -> %p\n", i, fn_name, (void *)iat[i].u1.Function);
            }
        } else {
            /* Count entries only */
            IMAGE_THUNK_DATA64 *iat = (IMAGE_THUNK_DATA64 *)(base + imp->FirstThunk);
            UINT cnt = 0; while (iat[cnt].u1.Function) cnt++;
            if (cnt > 0)
                fprintf(stderr, "  DLL: %-40s  (%u imports)\n", dll, cnt);
        }
    }
    fflush(stderr);
}

static int
InstallEscapeHook(HMODULE hNvcuda)
{
    void *real_escape = (void *)g_state.Escape;
    if (!real_escape) {
        fprintf(stderr, "IATHook: g_state.Escape not set\n"); return 0;
    }

    fprintf(stderr, "IATHook: scanning nvcuda64 IAT for D3DKMTEscape @ %p\n", real_escape);
    fflush(stderr);

    /* Dump gdi32-related imports to understand what nvcuda uses */
    DumpImports(hNvcuda, "nvcuda64");

    s_iat_entry = PatchIatEntry(hNvcuda, real_escape, (void *)EscapeHookFn);
    if (!s_iat_entry) {
        fprintf(stderr, "IATHook: D3DKMTEscape not found by address in nvcuda64 IAT\n"
                        "         trying by name in all import descriptors...\n");
        /* Try by name: scan all descriptors for any entry named D3DKMTEscape */
        uint8_t *base = (uint8_t *)hNvcuda;
        IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)base;
        IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)(base + dos->e_lfanew);
        DWORD imp_rva = nt->OptionalHeader
                           .DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]
                           .VirtualAddress;
        if (imp_rva) {
            IMAGE_IMPORT_DESCRIPTOR *imp = (IMAGE_IMPORT_DESCRIPTOR *)(base + imp_rva);
            for (; imp->Name && !s_iat_entry; imp++) {
                IMAGE_THUNK_DATA64 *orig = imp->OriginalFirstThunk ?
                    (IMAGE_THUNK_DATA64 *)(base + imp->OriginalFirstThunk) : NULL;
                IMAGE_THUNK_DATA64 *iat  = (IMAGE_THUNK_DATA64 *)(base + imp->FirstThunk);
                if (!orig) continue;
                for (UINT i = 0; orig[i].u1.AddressOfData && !s_iat_entry; i++) {
                    if (IMAGE_SNAP_BY_ORDINAL64(orig[i].u1.Ordinal)) continue;
                    IMAGE_IMPORT_BY_NAME *ibn = (IMAGE_IMPORT_BY_NAME *)
                        (base + (DWORD)orig[i].u1.AddressOfData);
                    if (_stricmp((char *)ibn->Name, "D3DKMTEscape") != 0) continue;
                    fprintf(stderr, "IATHook: found by name in %s[%u] -> %p\n",
                            (const char *)(base + imp->Name), i,
                            (void *)iat[i].u1.Function);
                    s_iat_entry = (void **)&iat[i].u1.Function;
                    s_iat_orig  = (void *)iat[i].u1.Function;
                    DWORD old_prot, dummy;
                    VirtualProtect(s_iat_entry, sizeof(void *), PAGE_READWRITE, &old_prot);
                    *s_iat_entry = (void *)EscapeHookFn;
                    VirtualProtect(s_iat_entry, sizeof(void *), old_prot, &dummy);
                }
            }
        }
        if (!s_iat_entry) {
            fprintf(stderr, "IATHook: not found — nvcuda uses GetProcAddress for D3DKMTEscape\n");
            fflush(stderr);
            return 0;
        }
    } else {
        s_iat_orig = real_escape;
    }
    s_hook_call_idx  = 0;
    fprintf(stderr, "IATHook: installed at IAT slot %p\n", s_iat_entry);
    fflush(stderr);
    return 1;
}

static void
RemoveEscapeHook(void)
{
    if (!s_iat_entry) return;
    DWORD old_prot, dummy;
    VirtualProtect(s_iat_entry, sizeof(void *), PAGE_READWRITE, &old_prot);
    *s_iat_entry = s_iat_orig;
    VirtualProtect(s_iat_entry, sizeof(void *), old_prot, &dummy);
    fprintf(stderr, "IATHook: restored  IAT slot %p (%u calls intercepted)\n",
            s_iat_entry, s_hook_call_idx);
    fflush(stderr);
    s_iat_entry = NULL;
    s_iat_orig  = NULL;
}

/* =========================================================================
 * Smoke test
 * ========================================================================= */

/* Retry KMT#2 with current g_state.hAdapter/hDevice, log result. */
static void retry_kmt2(const char *tag) {
    uint8_t buf2[sizeof(kKmt2_VersionCheck)];
    memcpy(buf2, kKmt2_VersionCheck, sizeof(buf2));
    D3DKMT_ESCAPE esc2 = {0};
    esc2.hAdapter              = g_state.hAdapter;
    esc2.hDevice               = g_state.hDevice;
    esc2.Type                  = D3DKMT_ESCAPE_DRIVERPRIVATE;
    esc2.Flags                 = 0;
    esc2.pPrivateDriverData    = buf2;
    esc2.PrivateDriverDataSize = sizeof(kKmt2_VersionCheck);
    esc2.hContext              = 0;
    LONG r = g_state.Escape(&esc2);
    fprintf(stderr, "%s KMT#2 retry: NTSTATUS=0x%08X%s\n", tag, (UINT)r, r==0?" OK":"");
    fflush(stderr);
}

int main(void)
{
    printf("opencuda Windows runtime — D3DKMTEscape path\n");

    if (opencuda_init() != 0) {
        fprintf(stderr, "opencuda_init failed\n");
        return 1;
    }
    printf("cuInit OK  adapter=0x%08X  device=0x%08X\n",
           g_state.hAdapter, g_state.hDevice);

    /* ---- Hook diagnostic ------------------------------------------------
     * Install an inline hook on gdi32!D3DKMTEscape, then run nvcuda64 cuInit.
     * The hook captures every escape — especially cmd=0x01000113 (KMT#2) —
     * so we can compare nvcuda64's actual payload against kKmt2_VersionCheck.
     * If the bytes differ, the blob is stale for this driver version.
     * If they match and nvcuda's call succeeds but ours fails, the issue is
     * something about adapter/device handle state before the call.
     * ------------------------------------------------------------------- */
    retry_kmt2("before-hook");
    {
        typedef int (*PFN_cuInit)(unsigned int);
        HMODULE hNvcuda = LoadLibraryA("nvcuda.dll");
        if (hNvcuda) {
            /* Install IAT hook AFTER nvcuda is loaded so we can scan its IAT */
            int hooked = InstallEscapeHook(hNvcuda);
            fprintf(stderr, "--- nvcuda64 cuInit with hook %s ---\n",
                    hooked ? "active" : "NOT installed (see above)");
            fflush(stderr);

            PFN_cuInit pcuInit = (PFN_cuInit)GetProcAddress(hNvcuda, "cuInit");
            if (pcuInit) {
                s_hook_call_idx = 0;
                int r = pcuInit(0);
                fprintf(stderr, "nvcuda.dll cuInit -> %d  (hook saw %u escapes)\n",
                        r, s_hook_call_idx);
            } else {
                fprintf(stderr, "nvcuda.dll: cuInit not found\n");
            }
            fflush(stderr);

            if (hooked) RemoveEscapeHook();
            FreeLibrary(hNvcuda);
        } else {
            fprintf(stderr, "nvcuda.dll load failed err=%lu\n", GetLastError());
        }
    }
    retry_kmt2("after-hook");

    if (opencuda_ctx_create() != 0) {
        fprintf(stderr, "opencuda_ctx_create failed\n");
        return 1;
    }
    printf("cuCtxCreate OK  ctx_handle=0x%08X\n", g_state.ctx_handle);

    /* Allocate 16 MB of GPU VRAM using the captured KMD private data blob. */
    D3DKMT_HANDLE handles[64] = {0};
    UINT          nhandles    = 0;
    int mem_rc = opencuda_mem_alloc(16 * 1024 * 1024, handles, &nhandles);
    if (mem_rc == 0) {
        printf("cuMemAlloc 16MB OK  nhandles=%u  h[0]=0x%08X\n", nhandles, handles[0]);
        /* TODO: call NtGdiDdDDIMapGpuVirtualAddress(handles, nhandles) -> gpu_va */
        opencuda_mem_free(handles, nhandles);
        printf("cuMemFree OK\n");
    } else if (mem_rc == -2) {
        printf("cuMemAlloc: D3DKMTCreateAllocation not found in gdi32\n");
    } else {
        printf("cuMemAlloc failed rc=%d NTSTATUS check log\n", mem_rc);
    }

    if (opencuda_ctx_destroy() != 0) {
        fprintf(stderr, "opencuda_ctx_destroy failed\n");
        return 1;
    }
    printf("cuCtxDestroy OK\n");

    return 0;
}
