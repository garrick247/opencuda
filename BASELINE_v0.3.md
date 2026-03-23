# Baseline: v0.3 — Structured Backend

**Tag:** `v0.3`
**Commit:** `54e595c136f4c4fd23c5b1df0eef2f66b456cb36`
**Date:** 2026-03-22

---

## Test Suite

| Metric | Value |
|--------|-------|
| Total tests | 280 |
| Passing | 280 |
| Failing | 0 |
| Xfailed | 0 |

Test files: `test_compiler.py`, `test_ptx_inspect.py`, `test_invariants.py`, `test_error_cases.py`

---

## Register Allocation Benchmark (51 kernels)

Run: `python tools/benchmark_regalloc.py`

| Metric | Value |
|--------|-------|
| Kernels measured | 51 |
| Average reduction vs naive SSA | **1.74×** |
| Best reduction (matmul_tiled) | **4.36×** |
| Average gap ratio | **1.01** |
| Max instructions (matmul_tiled) | 302 |

**Gate:** Future changes must not degrade average reduction below **1.5×** or average gap ratio above **1.15**.

Selected kernel benchmarks:

| Kernel | Instructions | Naive | Declared | Reduction | Gap |
|--------|-------------|-------|----------|-----------|-----|
| matmul_tiled | 302 | 257 | 59 | 4.36× | 1.00 |
| register_pressure | 82 | 86 | 30 | 2.87× | 1.00 |
| bitwise_test | 30 | 34 | 14 | 2.43× | 1.00 |
| chain_reuse | 21 | 22 | 9 | 2.44× | 1.00 |
| reduce | 66 | 37 | 21 | 1.76× | 1.00 |
| vector_add | 24 | 20 | 14 | 1.43× | 1.00 |

---

## Supported Language Features

| Feature | Status | Notes |
|---------|--------|-------|
| `float` / `int` / `unsigned` / `double` | Full | All arithmetic ops |
| `float16` / `half` | Full | Native `.f16` PTX, `0h` hex constants |
| Pointers + array indexing | Full | 64-bit addressing throughout |
| `__global__` kernels | Full | |
| `__device__` functions | Full | Inlined at call site |
| Multi-return `__device__` | Full | `return expr;` inside if-blocks |
| `__shared__` memory | Full | |
| `__syncthreads()` | Full | `bar.sync 0` |
| `threadIdx` / `blockIdx` / `blockDim` | Full | All `.x/.y/.z` |
| `atomicAdd` / `atomicCAS` | Full | |
| Warp shuffles (`__shfl_sync` etc.) | Full | |
| `printf` / `vprintf` ABI | Full | Stack valist, global format strings |
| `__ldg()` | Full | `ld.global.nc` |
| `__restrict__` | Parsed | Silently stripped |
| `#define NAME VALUE` | Full | |
| `for` / `while` / `do-while` | Full | |
| `break` / `continue` | Full | |
| `switch` | Full | |
| Structs (value type) | Full | Field access, no pointers-to-struct |
| Local arrays | Full | `.local` memory |
| Ternary operator | Full | |
| Loop unrolling (≤16 trip count) | Full | Compile-time constant bounds only |
| Constant folding | Full | With strength reduction (mul→shl) |
| CSE | Full | Per-basic-block, type-aware |

### Known Limitations (v0.3)

1. `float16` emits as `f32` in mixed-promotion contexts where `half` operands are widened
2. Device function inlining does not support multiple `return` points in nested if-else (one level works)
3. Integer division/remainder emits PTX `div`/`rem`; relies on ptxas for SASS expansion
4. Register allocation is linear scan, no graph coloring; no spilling (kernels must fit in register file)
5. No texture/surface memory
6. No cooperative groups
7. No tensor operations (wmma/mma)
8. `&&` / `||` evaluate both sides (no short-circuit)
9. `++i` / `--i` prefix increment not supported (ParseError)
10. Recursive `__device__` functions not supported (RecursionError at compile time)
11. `printf` requires string literal format argument
12. No multi-file compilation; single `.cu` input only
13. `float16` loads use `.b16` not `.f16` (PTX requirement; transparent to user)
14. Struct pointers and pointer-to-struct not supported
15. No variadic device functions

---

## Regression Gates for Future Work

Before merging any change:

1. `pytest opencuda/tests/ -q` → **280 passed, 0 failed**
2. `python tools/benchmark_regalloc.py` → avg reduction **≥ 1.5×**, avg gap **≤ 1.15**
3. No new PTX structural assertion failures in `test_ptx_inspect.py`
