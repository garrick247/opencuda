// Probe: half precision comparison returning bool-style int (__hbeq, __hbne, __hbgt,
// __hblt, __hbge, __hble), __hequ/__hneu/__hgtu/__hltu (unordered comparisons),
// __hmax3/__hmin3 (three-arg), warp reduce on half via float conversion,
// __hfma2 (if supported), and stress test of half in conditional branching.

// ------------------------------------------------------------------
// __hbeq / __hbne / __hbgt / __hblt / __hbge / __hble (bool-returning half cmp).

__global__ void half_bool_cmp(int *out, unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        int eq = __hbeq(ha, hb);
        int ne = __hbne(ha, hb);
        int gt = __hbgt(ha, hb);
        int lt = __hblt(ha, hb);
        int ge = __hbge(ha, hb);
        int le = __hble(ha, hb);
        // Encode: eq|ne|gt|lt|ge|le
        out[tid] = (eq<<5)|(ne<<4)|(gt<<3)|(lt<<2)|(ge<<1)|le;
    }
}

// ------------------------------------------------------------------
// Unordered half comparisons: __hequ / __hneu / __hgtu / __hltu.

__global__ void half_unordered_cmp(int *out, unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        int equ = __hequ(ha, hb);
        int neu = __hneu(ha, hb);
        int gtu = __hgtu(ha, hb);
        int ltu = __hltu(ha, hb);
        out[tid] = (equ<<3)|(neu<<2)|(gtu<<1)|ltu;
    }
}

// ------------------------------------------------------------------
// Half in conditional: use __hgt to branch.

__global__ void half_branch(unsigned short *out, unsigned short *a, unsigned short *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half r;
        if (__hgt(ha, hb)) {
            r = __hmul(ha, __float2half(2.0f));
        } else {
            r = __hadd(hb, __float2half(1.0f));
        }
        out[tid] = __half_as_ushort(r);
    }
}

// ------------------------------------------------------------------
// Half loop: accumulate sum of half array using __hadd.

__global__ void half_accum(float *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half s = __float2half(0.0f);
        for (int i = 0; i < 8 && (tid*8+i) < n; i++) {
            __half h = __ushort_as_half(in[tid*8 + i]);
            s = __hadd(s, h);
        }
        out[tid] = __half2float(s);
    }
}

// ------------------------------------------------------------------
// __hfma variants.

__global__ void hfma_variants(unsigned short *out_basic,
                                unsigned short *out_sat,
                                unsigned short *a, unsigned short *b,
                                unsigned short *c, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        __half hc = __ushort_as_half(c[tid]);
        out_basic[tid] = __half_as_ushort(__hfma(ha, hb, hc));
        out_sat[tid]   = __half_as_ushort(__hfma_sat(ha, hb, hc));
    }
}

// ------------------------------------------------------------------
// __habs2 (if available) — fall back to __habs if not.

__global__ void half_abs_chain(unsigned short *out, unsigned short *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half v = __ushort_as_half(in[tid]);
        __half a = __habs(v);
        __half b = __hneg(a);   // negate the abs
        __half c = __habs(b);   // abs of neg(abs) = abs
        out[tid] = __half_as_ushort(c);
    }
}

// ------------------------------------------------------------------
// Half with ternary selector.

__global__ void half_ternary(unsigned short *out, unsigned short *a,
                               unsigned short *b, int *sel, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        __half ha = __ushort_as_half(a[tid]);
        __half hb = __ushort_as_half(b[tid]);
        // Use __hgt as ternary condition
        __half r = __hgt(ha, hb) ? ha : hb;
        out[tid] = __half_as_ushort(r);
    }
}
