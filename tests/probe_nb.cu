// Probe: variable name collision between inline locals and caller variables
// The parser uses a flat _variables dict — inline local 'r' must not
// clobber caller's variable 'r' when both exist in scope.

struct S { float x; int n; };
struct P { float a, b; };

// Inline uses local 'r' (common pattern from "S r; ... return r;")
__device__ S make_s(float x, int n) {
    S r;
    r.x = x;
    r.n = n;
    return r;
}

__device__ P make_p(float a, float b) {
    P r;
    r.a = a;
    r.b = b;
    return r;
}

// Caller also has a variable named 'r' — must stay isolated from inline's 'r'
// Expected: r.x=0, r.n=0 unchanged; result.x=in[0], result.n=n
__global__ void name_collision_r(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        S r;
        r.x = 0.0f;
        r.n = 0;
        S result = make_s(in[0], n);  // inline also uses 'r' locally
        out[0] = r.x;          // must be 0.0f (not in[0])
        out[1] = (float)r.n;   // must be 0   (not n)
        out[2] = result.x;     // must be in[0]
        out[3] = (float)result.n;  // must be n
    }
}

// Caller has 'r' in a loop, inline uses 'r' each iteration
// Expected: r.x and r.n accumulate; result each iter is make_s(in[i], i)
__global__ void collision_in_loop(float *out, float *in, int n) {
    int tid = threadIdx.x;
    if (tid == 0) {
        S r;
        r.x = 0.0f;
        r.n = 0;
        P acc;
        acc.a = 0.0f;
        acc.b = 0.0f;
        for (int i = 0; i < n; i++) {
            // make_p also uses local 'r' internally → same collision risk
            P p = make_p(in[i], in[i] * 2.0f);
            acc.a += p.a;
            acc.b += p.b;
            r.x += 1.0f;   // r must not be clobbered by make_p's local 'r'
            r.n++;
        }
        out[0] = r.x;    // must be (float)n
        out[1] = (float)r.n;  // must be n
        out[2] = acc.a;
        out[3] = acc.b;
    }
}
