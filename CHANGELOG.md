# Changelog

## v0.3 — Structured Backend (2026-03-22)

### New Features

**float16 (half-precision)**
- Native `.reg .f16` register class with prefix `h`
- `add.f16` / `sub.f16` / `mul.f16` / `div.approx.f16` instructions
- `ld.global.b16` / `st.global.b16` (PTX requires `.b16` not `.f16` for load/store)
- Half constants emitted as `0h<4hexdigits>` via IEEE 754 bit-packing
- `float16` / `half` type keyword in source; edge values and mixed-type arithmetic tested

**Multi-return device functions**
- `return expr;` inside `if`/`else` blocks inside inlined `__device__` functions
- `_inline_return_target` mechanism: return statements redirect to a merge block,
  dual-write to single physical dest register (safe: branches are mutually exclusive)
- Tested: `device_return.cu`, `multi_return.cu`, `nested_returns.cu`, `inline_printf_return.cu`

**printf / vprintf lowering**
- Full PTX vprintf ABI: `.local .align 8 .b8 _valist_N[8*nargs]` on stack
- Format string → `.global .align 1 .b8 _fmt_N[...]` at file scope
- `cvta.local.u64` for generic pointer; `cvt.u64.u32` for int args; `cvt.f64.f32` for float args
- `call.uni (retval), vprintf, (fmt_ptr, valist_ptr)` with proper `.param` blocks
- `__preamble__` key in `ir_to_ptx()` carries extern declaration + all format strings

**`__ldg()` / `__restrict__`**
- `__ldg(ptr)` → `ld.global.nc.{type}` (read-only cache load)
- `__restrict__` qualifier silently stripped (no semantic effect in current IR)
- Tested: `ldg_test.cu`, `ldg_arithmetic.cu`

**Linear scan register allocation**
- Live interval computation across all instructions in flattened basic block order
- Per-type-prefix free-list reuse (r / rd / f / fd / h / p buckets)
- Fallback allocator (`_fallback_alloc`, `_fallback_count`) for emission-time temporaries
- Widen-cache bug fixed: no longer uses raw SSA IDs as physical register indices
- Avg reduction across 51 kernels: **1.74×** vs naive SSA allocation; gap ratio avg 1.01

### Tests

**New test files:**
- `opencuda/tests/test_ptx_inspect.py` — 22 structural PTX assertion tests (f16 regs,
  vprintf layout, `ld.global.nc`, multi-return branch structure, compact register numbering,
  liveness/reuse patterns across `branch_overlap`, `chain_reuse`, `merge_reuse`)
- `opencuda/tests/test_invariants.py` — 8 compiler invariant tests: every referenced register
  index within declared count (parametrized over all 51 kernels), linear scan reduces b32/f32
  vs naive SSA (parametrized), gap ratio assertions, alloc map coverage, widen-cache correctness
- `opencuda/tests/test_error_cases.py` — Intentional failure tests: `++i` raises ParseError,
  undefined variable raises, missing semicolon raises, printf non-literal degrades cleanly,
  recursive device function hits limit, void return in global kernel works, empty kernel valid

**New test kernels (liveness torture):**
- `tests/branch_overlap.cu` — value live across both if/else branches
- `tests/merge_reuse.cu` — phi-like pattern at join point
- `tests/chain_reuse.cu` — chain of short-lived values, should reuse ≤4 f32 registers
- `tests/mixed_type_pressure.cu` — int / float / ptr all simultaneously live

### Tools

- `tools/benchmark_regalloc.py` — register count, instruction count, reduction factor,
  gap ratio across all real kernels; `--sort`, `--min-instructions`, `--csv` flags

### Documentation

- `docs/language_spec.md` — 650-line truth-based language specification: types, operators,
  statements, CUDA built-ins, preprocessor, functions, structs, optimization passes,
  PTX target details, 15 known limitations

### Bug Fixes

- **Widen-cache raw-ID bug**: `_reg_counts[prefix] = max(..., wide.id + 1)` used the SSA value
  ID (e.g. 20) as a physical register index, producing `.reg .b64 %rd<21>` for `vector_add`.
  Fixed by routing all widen-cache registers through the fallback allocator.
- **test_invariants multi-kernel false positive**: `_extract_reg_decls` overwrote instead of
  taking max, causing multi-kernel files to fail with stale per-kernel declaration counts.

---

## v0.2 — CVT CSE (2026-03-17)

- Eliminate redundant `cvt.u64.u32` instructions via widen cache in codegen
- `_widen_cache` tracks previously widened 32-bit registers to avoid re-emission

---

## v0.1 — Initial Release (2026-03-16)

- Pure-Python CUDA-subset C → PTX 9.0 compiler targeting SM_120 (Blackwell / RTX 5090)
- 33 test kernels covering: vector ops, matrix multiply (naive + tiled), reduction, stencil,
  histogram, shared memory, warp shuffles, atomics, structs, loops, conditionals
- Constant folding with strength reduction (mul → shl for power-of-2 constants)
- Common Subexpression Elimination (per-basic-block, type-aware)
- Loop unrolling for compile-time trip counts ≤ 16
- GPU cubin loader/tester for hardware verification against RTX 5090
