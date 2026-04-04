# OpenCUDA — Architecture Guide

Internal reference for contributors. The README covers the user-facing story; this covers
what you need to know to safely modify the compiler.

---

## Pipeline Overview

```
.cu source
  └─ preprocess()          frontend/preprocess.py   — strip comments, handle #define/#include
      └─ parse()           frontend/parser.py        — recursive descent → IR nodes
          └─ optimize()    ir/optimize.py            — 13-pass optimizer pipeline
              └─ emit()    codegen/emit.py            — IR → PTX text
```

All stages operate on a `Module` (ir/nodes.py), which contains a list of `Kernel` objects.
Each `Kernel` is a CFG of `BasicBlock` objects. Each block holds a list of instruction
objects and a terminator.

---

## IR Nodes (ir/nodes.py)

Every IR value is a `Value(name, ty, id)`. IDs are globally unique integers. Constants are
`Const(value, ty)`. The union type `Operand = Value | Const` appears everywhere.

### Instruction types

| Class | Fields | Notes |
|-------|--------|-------|
| `BinInst` | dest, lhs, rhs, op (BinOp) | Arithmetic, bitwise, shift |
| `CmpInst` | dest, lhs, rhs, op (CmpOp) | Result type is always BOOL |
| `SelectInst` | dest, cond, true_val, false_val | Predicated select (ternary) |
| `CvtInst` | dest, src | Type conversion |
| `LoadInst` | dest, addr | Load from pointer |
| `StoreInst` | addr, value | Store to pointer (no dest) |
| `CallInst` | dest, func, args | Built-ins + device functions |
| `ParamInst` | dest | Kernel parameter binding |
| `PhiInst` | dest, incoming (list of (val, label)) | Rare; loop writeback pattern |
| `PrintfInst` | fmt, args | CUDA printf |
| `GlobalAddrInst` | dest, name | Address of .global symbol |
| `AsmInst` | constraints, asm_str, inputs, outputs | Inline PTX asm |

### Terminators

| Class | Fields |
|-------|--------|
| `BrTerm` | target (label) |
| `CondBrTerm` | cond (Value), true_bb, false_bb |
| `RetTerm` | ret_val |

### Types (ir/types.py)

`ScalarTy`, `PtrTy(pointee_ty, addr_space)`, `VoidTy`. Address spaces: `GLOBAL`, `SHARED`,
`LOCAL`, `PARAM`, `GENERIC`. The addr_space on a `PtrTy` determines which PTX `.space`
qualifier gets emitted.

---

## Optimizer Passes (ir/optimize.py)

Passes run in a fixed order. The key insight is that many passes expose opportunities for
others, so several groups run in fixpoint loops.

### Pass order

```
1.  constant_fold          — fold BinInst/CmpInst/CvtInst with two Const operands
2.  copy_propagation       — eliminate add/cvt identity chains (x+0, cvt(x) where ty==x.ty)
3.  cse                    — deduplicate identical computations; cross-block for &&/||
4.  identity_fold          — x+0, x*1, x<<0, etc. → x
5.  dead_inst_elim         — remove instructions whose dest is never used
6.  dead_block_elim        — remove unreachable blocks (BFS from entry)
7.  licm                   — hoist loop-invariant loads/computes out of loop headers
8.  unroll_loops           — unroll small loops (≤16 iters, constant bound)
9.  constant_fold  (2nd)   — fold constants exposed by unrolling
10. identity_fold  (2nd)
11. dead_inst_elim (2nd)
12. thread_empty_blocks → [dead_block_elim + dead_inst_elim + thread_empty_blocks] to fixpoint
    — thread_empty_blocks bypasses unconditional branches through transparent blocks
    — fixpoint needed: dead_inst_elim can make a block transparent, enabling more threading
13. post-thread fixpoint   — [unroll + cf + cse + idf + die + dbe + teb] × 8
    — after threading, loops become recognizable for unrolling; new CSE/fold opportunities appear
```

**Why the two fixpoints matter:** After `thread_empty_blocks`, previously hidden loop
structures become visible to `unroll_loops`, and newly linear chains expose CSE hits. A
single post-thread pass is not idempotent; the fixpoint (capped at 8 rounds) ensures
`optimize(optimize(x)) == optimize(x)`.

### CSE cross-block propagation

Within a block, CSE tracks seen computations in a `seen: dict[key, Value]`. Normally this
dict is fresh per block. The cross-block extension: if a block has exactly one predecessor
AND that predecessor has already been processed, the current block inherits the predecessor's
final `seen` dict.

**Why it's safe:** A single-predecessor block is dominated by its predecessor. Any value
defined in the predecessor is defined on all paths to the current block.

**When to be careful:** This only propagates pure computations (BinInst, CmpInst, CvtInst,
pure CallInst). LoadInst results are never put into `seen` — a load's result isn't a pure
function of its operands across block boundaries.

**Multi-definition values are skipped:** Loop writeback values (e.g., the loop counter)
have `def_count >= 2`. These are intentionally exempt from CSE — merging them would alias
loop variables and corrupt semantics.

### Ternary diamond (codegen/emit.py)

CUDA ternary `cond ? a : b` often compiles to a "diamond" CFG pattern:
```
      entry → tern_true → tern_merge
           ↘ tern_false ↗
```

The emitter detects this pattern and emits inline predicated moves rather than BRA-based
divergence:
```ptx
setp.eq.s32 %p0, %cond, 1
@%p0  mov.f32 %r0, %true_val
@!%p0 mov.f32 %r0, %false_val
```

**Predicate-as-value edge case:** Cross-block CSE can make the "value" operand a predicate
register (BOOL type). PTX does not allow `mov.s32 %r, %p`. The emitter detects this
(`v.id in self._pred_ids`) and uses `selp.type dest, 1, 0, %pred` instead.

---

## IR Verifier (ir/verify_ir.py)

`verify_kernel(kernel)` returns a list of error strings. Run after every optimization pass
during testing. Seven checks:

1. **Terminator presence** — every block must have a non-None terminator
2. **Branch target validity** — every branch target must name an existing block
3. **Block reachability** — every block reachable from entry (disable with `check_reachability=False`)
4. **Def presence** — every used Value must have a defining instruction; exception: `PtrTy(SHARED/LOCAL)` values are exempt (defined by PTX `.shared` declaration, not an IR instruction)
5. **Domination** — for single-definition Values, the defining block must dominate all use sites; multi-definition Values (loop writeback, def_count ≥ 2) are skipped
6. **Critical edges** — edges from multi-successor to multi-predecessor blocks (reported as warnings)
7. **Single-entry loop guarantee** — every loop header must dominate all reachable blocks in its body

---

## Codegen (codegen/emit.py)

The `Emitter` class walks the optimized kernel and emits PTX text. Key design points:

**Register allocation** is per-type and per-function. PTX types map to register prefixes:
`%r` (s32), `%rd` (s64/u64), `%f` (f32), `%fd` (f64), `%p` (pred). The emitter scans all
instructions first to collect which Values exist, assigns them register names, then emits
declarations followed by the body.

**Shared memory** is emitted as `.shared .align N .b8 name[size]` declarations at the top
of each kernel. The allocation size comes from parser annotations; the IR holds only a
pointer Value whose operand is the base address.

**`__syncthreads()`** emits `bar.sync 0;` inline.

**Device function inlining:** Small device functions are inlined at call sites. The inliner
renames Values to avoid collisions and splices the callee's blocks into the caller's CFG.

---

## Known Pre-existing Failures (37 total)

These were present before v0.9 and are tracked but not fixed. They do **not** affect
correctness of the PTX output for well-formed kernels.

| Category | Count | Root cause |
|----------|-------|------------|
| `test_verifier.py` — struct field access | ~20 | Struct fields accessed from shared memory produce Values without defining IR instructions. The verifier correctly flags them; PTX zero-initializes registers, masking the issue at runtime. |
| `test_verifier.py` — inline asm | ~7 | `AsmInst` outputs are Values that don't have defining IR instructions in the conventional sense. |
| `test_dominance.py` — probe_lj | 4 | Encoding edge case in the dominance tree for a specific CFG pattern. |
| `test_mem_invariants.py` — probe_z* | 4 | Parameter count mismatch in kernels with device function pointer params. |
| `test_error_cases.py` | 2 | Prefix increment and recursive device function error paths not yet implemented. |

**Do not add new tests to `KNOWN_PARSER_BUGS` without also filing a bug.** The set is
currently empty because all known parser bugs were fixed in v0.51; new entries mean a
regression.

---

## Test Structure

```
opencuda/tests/
  test_compiler.py         — parse + emit + ptxas validates (end-to-end)
  test_verifier.py         — IR verifier checks after each pass
  test_opt_legality.py     — optimize(optimize(x)) == optimize(x) idempotency
  test_cse_legality.py     — CSE-specific correctness checks
  test_licm_legality.py    — LICM-specific correctness checks
  test_invariants.py       — register type invariants, linear scan
  test_dominance.py        — dominator tree correctness
  test_mem_invariants.py   — memory instruction parameter invariants
  test_error_cases.py      — expected parse/compile errors

tests/                     — .cu source files used as test inputs
  probe_*.cu               — targeted probes for specific language features
  gpu_*.cu                 — GPU execution tests (require hardware)
```

Run the full suite: `pytest opencuda/tests/ -x -q`

Run only compiler tests: `pytest opencuda/tests/test_compiler.py -q`

---

## Adding a New IR Instruction

1. Add the class to `ir/nodes.py` following the dataclass pattern
2. Add operand enumeration to `_operand_values()` in `ir/verify_ir.py`
3. Handle it in the optimizer's inline replacement loops in `ir/optimize.py` (both the
   per-block replacement and the global second-pass replacement)
4. Emit PTX in `codegen/emit.py`
5. Add a probe test in `tests/`

If the instruction has a `dest` Value, the verifier will automatically track its definition.
If it has side effects (memory writes, barriers), do NOT add it to the CSE `seen` dict.

---

## Common Pitfalls

**Touching optimize.py's pass order:** The idempotency tests (`test_opt_legality.py`,
`test_cse_legality.py`, `test_licm_legality.py`) will catch regressions. Run them before
pushing any optimizer change. The fixpoint loops in passes 12 and 13 exist specifically
because single-pass cleanup was not idempotent — don't simplify them away.

**Modifying CSE's `seen` propagation:** The single-predecessor guard is load-bearing. If
you extend propagation to multi-predecessor blocks (e.g., a basic loop join), you must
verify that the canonical Value dominates the use site. The verifier's domination check
(check 5) will catch violations, but only if `def_count == 1`.

**Shared memory Values in the verifier:** Values of type `PtrTy(AddrSpace.SHARED)` or
`PtrTy(AddrSpace.LOCAL)` are exempt from the "undefined value" check. If you change how
shared memory is represented in the IR, update `_is_smem()` in `verify_ir.py` or the
verifier will produce false positives.
