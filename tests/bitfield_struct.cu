// Regression: struct with bitfields — unsigned int x : 16;
// Without fix: COLON after field name → ParseError "expected SEMI, got COLON ':'".
// Fix: _parse_struct_def checks for COLON after field name; if found, consumes
//   the bitfield width expression (the width is ignored — all fields get their
//   base type's size for layout purposes, which is sufficient for PTX codegen).

typedef struct {
    unsigned int x : 16;
    unsigned int y : 16;
} PackedCoord;

typedef struct {
    unsigned int flags : 8;
    unsigned int id    : 20;
    unsigned int valid : 4;
} PackedDesc;

typedef struct {
    int value;
    unsigned int tag : 3;
    unsigned int pad : 29;
} TaggedInt;

__global__ void bitfield_test(PackedCoord *coords, int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        // Access fields normally (bitfield layout not enforced in PTX,
        // but parsing must not crash)
        out[tid] = (int)coords[tid].x + (int)coords[tid].y;
    }
}

__global__ void tagged_test(TaggedInt *items, int *out, int n) {
    int tid = threadIdx.x;
    if (tid < n) {
        out[tid] = items[tid].value;
    }
}
