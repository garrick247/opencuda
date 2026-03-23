# Changelog

## v0.6 — Optimization Legality (2026-03-22)

### New Optimization Passes

**Dead Block Elimination (`dead_block_elim`)**
- BFS from entry block; removes all unreachable basic blocks
- Eliminates `after_break_*` and `after_continue_*` stub blocks produced by loop lowering
- Runs before `identity_fold` so dead-block definitions don't inflate def_count

**Identity Fold / Copy Propagation (`identity_fold`)**
- Propagates `add D, V, Const(0)` for single-definition Values (def_count == 1)
- Chain-follows aliases: `D = V + 0; E = D + 0` → both resolve to `V`
- Propagates into BinInst, CmpInst, LoadInst, StoreInst, CvtInst operands and CondBrTerm condition
- **Safety guarantee**: loop-writeback Values have def_count ≥ 2 and are never touched

**Dead Instruction Elimination (`dead_inst_elim`)**
- Iterates to fixpoint; removes BinInst/CmpInst/CvtInst whose dest is never consumed
- Collects used Value IDs from all instruction sources and CondBrTerm.cond
- Does not touch StoreInst, LoadInst, PrintfInst, or CallInst (side-effectful)

**Pass ordering**: unroll → constant_fold → cse → dead_block_elim → identity_fold → dead_inst_elim

### Tests

**New test file:**
- `opencuda/tests/test_opt_legality.py` — 413 tests across 10 test groups:
  - Structural/CFG/memory invariants hold after optimization (parametrized, all .cu files)
  - Idempotency: `optimize()` twice produces identical PTX
  - Dead stub blocks absent from optimized PTX
  - `dead_block_elim` API: removes synthetic unreachable blocks, preserves reachable ones
  - `identity_fold` removes add-zero copies without corrupting loop-carried Values
  - `dead_inst_elim` API: removes unused BinInst/CmpInst/CvtInst
  - Loop correctness after all passes: for/while writeback still functional
  - Known kernels: optimized PTX ≤ raw instruction count (no inflation)
  - Pass interaction: dead_block_elim+identity_fold order verified correct

### Bug Fixes

- **`identity_fold` AttributeError on None dest**: `hasattr(inst, 'dest')` matched instruction
  types where `dest` is `None` (e.g. certain store variants). Fixed with `inst.dest is not None`
  guard.

### Metrics (72 kernels)

- Avg reduction 1.74× (same as v0.3 — new passes clean up structural noise, not register pressure)
- Avg gap ratio 1.01 — same tight packing
- All 1240 tests pass, 13 skipped

---

## v0.5 — Memory + ABI Hardening (2026-03-22)

### New Features

**Memory torture kernel suite** (`tests/nasty_mem_*.cu`, 10 kernels)
- Scatter writes, mixed-type loads, local scratch arrays, predicated stores, pointer arithmetic,
  loop-carried stores, conditional store merging, multi-pointer address-space, widen/narrow casts,
  device function call with many pointer args

**Benchmark expansion** (`tools/benchmark_regalloc.py`)
- Added `count_loads()`, `count_stores()`, `count_branches()`, `count_converts()`
- New table columns: `Lds`, `Sts`, `Brs`, `Cvt`

### Tests

**New test files:**
- `opencuda/tests/test_mem_inspect.py` — 10 structural PTX assertions per kernel:
  pointer params use u64, half uses b16, address regs are %rd, .local opcode, predicated
  guard before conditional stores, add.u64 for pointer arithmetic, store position after merge
- `opencuda/tests/test_mem_invariants.py` — 5 invariants parametrized over all 65 kernels:
  no f16 opcodes in params/stores, 64-bit address regs on ld/st.global, no cross-width float stores

---

## v0.4 — Control-Flow Hardening (2026-03-22)

### New Features

**While/do-while loop writeback fix**
- Condition variable mutations now feed back into loop condition on every iteration
- `_loop_writeback()` helper: snapshot vars before loop, emit `add entry_val, cur_val, 0`
  writeback at end of body, restore `_variables` to entry values
- Fixed symmetrically for both `while` and `do-while` loops

**Half-precision PTX correctness**
- `0h####` immediates not accepted by ptxas in `add.f16`; now materialized via `cvt.rn.f16.f32`
- `st.global.f16` → `st.global.b16`; `ld.param.f16` → `ld.param.b16`;
  `.param .f16` → `.param .b16` in kernel signatures
- `cvt.rn.f64.f32` → `cvt.f64.f32` (no `.rn` for exact widening)

**Mixed-type float coercion** (`_coerce_to_float()`)
- Emits cvt before arithmetic when int or half operand used in f32/f64 instruction
- Handles: f16→f32, s32/u32→f32, f32→f64, f16→f64 coercions

**`_result_type` float promotion fix**
- `half + half` → HALF (previously incorrectly returned FLOAT)
- Wider float wins: `half + float` → FLOAT, `float + double` → DOUBLE

**Nasty kernel suite** (`tests/nasty_*.cu`, 9 kernels)
- `nasty_nested_break`, `nasty_early_exit`, `nasty_while_update`, `nasty_loop_half`,
  `nasty_branch_widen`, `nasty_ldg_cond`, `nasty_printf_guard`, `nasty_multi_exit`,
  `nasty_type_cast_loop`

### Tests

**New test file:**
- `opencuda/tests/test_cfg.py` — 59 tests:
  - bra target existence, no double terminators, reachability, RetTerm reachable
  - While writeback: condition variable written back each iteration
  - Nested break isolation: inner break targets inner exit
  - Multi-exit convergence: all break paths hit same exit block
  - Early return inside loop does not corrupt inc_bb
  - Structural/quality parametrized over all nasty_*.cu files

---

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
