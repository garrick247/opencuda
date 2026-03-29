// Probe: Unusual type combinations and casts in expressions
// - Cast in binary expression: (int)a + (int)b
// - Cast in comparison: (float)int_var < float_limit
// - Cast of array element: (float)arr[i]
// - Integer division with cast for floating result: (float)a / (float)b

__global__ void integer_division_as_float(float *out, int *a, int *b, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Integer division (truncating): a[tid] / b[tid]  
        int int_div = (b[tid] != 0) ? a[tid] / b[tid] : 0;
        // Float division: (float)a / (float)b
        float flt_div = (b[tid] != 0) ? (float)a[tid] / (float)b[tid] : 0.0f;
        out[tid] = flt_div + (float)int_div;
    }
}

// Indexed array with casted subscript
__global__ void cast_subscript(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        unsigned int utid = (unsigned int)tid;
        float v = in[(int)utid];  // cast of unsigned to int in subscript
        out[tid] = (float)(int)(v + 0.5f);  // round-to-int via cast
    }
}

// Mixed type arithmetic chain
__global__ void mixed_chain(float *fout, double *dout, int *iin, float *fin, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        int iv = iin[tid];
        float fv = fin[tid];
        double dv = (double)fv;
        float mixed = (float)iv * fv + (float)dv;
        double dmixed = (double)iv * dv + (double)mixed;
        fout[tid] = mixed;
        dout[tid] = dmixed;
    }
}
