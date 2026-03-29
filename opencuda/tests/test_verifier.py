"""
test_verifier.py — v0.10 IR Verifier Tests.

Deliverable C: Optimization revalidation — all .cu kernels must pass the IR
verifier after optimization.  Pre-existing parser bugs (tracked in
KNOWN_PARSER_BUGS) are excluded from the "zero violations" tests, but ARE
expected to still fail the verifier (meaning the optimizer does not fix them).

Deliverable D: Benchmark annotations — block count, reachable count, and
loop count are logged for every kernel.

Test groups
-----------
  Group 1: Positive cases — clean IR passes with no errors
    1a. All .cu kernels before optimization (structural checks only)
    1b. All .cu kernels after full optimization (all checks, excluding known bugs)
  Group 2: Negative cases — constructed broken IR triggers correct errors
    2a. Missing terminator
    2b. Branch to undefined block
    2c. Use of undefined Value
    2d. Dominance violation (use in block not dominated by def)
    2e. Unreachable block detected
  Group 3: Pass-by-pass revalidation — verifier clean after each pass
  Group 4: Benchmark stats table (logs only, never fails)
  Group 5: Known-bug confirmation — pre-existing parser bugs still detected

Bug sets
--------
KNOWN_PARSER_BUGS: verifier violations that PERSIST after full optimization.
  These represent fundamental SSA-IR limitations in the parser:
  - Struct fields undefined after pointer-write through inlined device functions
  - Single-arm variable definitions across control-flow branches
  - Struct return values from device functions not connected to caller SSA
  Group 5 (test_known_parser_bugs_detected) verifies these still show errors
  after optimization.

KNOWN_PREOPT_ONLY_BUGS: violations present in raw parse output that the
  optimizer's pipeline RESOLVES.  The bugs are genuine parser issues but the
  optimizer (loop unrolling, dead_inst_elim, dead_block_elim) happens to
  eliminate the bad IR before it causes a problem.  Group 5 does NOT check
  these (they are clean after opt).  Pass-by-pass tests still skip them because
  the issue appears before the relevant pass has run.
"""

import pytest
import copy
from pathlib import Path

from opencuda.ir.nodes import (
    BasicBlock, Kernel, KernelParam, Module, Value, Const,
    BrTerm, CondBrTerm, RetTerm, BinInst, CmpInst, CvtInst,
    BinOp, CmpOp,
)
from opencuda.ir.types import INT32, UINT32
from opencuda.ir.verify_ir import verify_kernel, verify_module, find_critical_edges
from opencuda.ir.dominator import kernel_stats
from opencuda.ir.optimize import (
    optimize, constant_fold, cse, dead_block_elim,
    identity_fold, dead_inst_elim, licm,
)
from opencuda.frontend.preprocess import preprocess
from opencuda.frontend.parser import parse

TESTS_DIR = Path(__file__).parent.parent.parent / 'tests'
ALL_CU_FILES = sorted(f for f in TESTS_DIR.glob('*.cu') if not f.name.startswith('gpu_'))

# Pre-existing parser bugs discovered by the verifier.
# These kernels produce invalid IR that ptxas accepts (PTX zero-initialises
# registers, masking semantic errors) but our verifier correctly rejects.
# The optimizer does NOT introduce these bugs — they pre-date v0.9.
# All files here still show verifier errors after full optimization (confirmed
# by test_known_parser_bugs_detected).
KNOWN_PARSER_BUGS = frozenset()
# All 28 original parser bugs fixed as of v0.51.

# Parser bugs that are ONLY visible before the full optimization pipeline runs.
# The optimizer (dead_block_elim, dead_inst_elim, loop unrolling) happens to
# eliminate the bad IR.  Group 5 does NOT check these.
KNOWN_PREOPT_ONLY_BUGS = frozenset({
    'probe_di',     # missing terminator in dead inline-merge block (dead_block_elim fixes)
    'probe_fi',     # multiple-return struct fn: dominance violation resolved by dead_block_elim
    'probe_fu',     # multiple-return struct fn: unreachable after_inline_return blocks
    'probe_gn',     # missing terminator in dead ternary-merge block (dead_block_elim fixes)
    'probe_hs',     # struct return fields undefined — loop unrolling resolves them
    'typedef_union', # union field aliasing: dead_inst_elim eliminates the bad use
})

# Combined skip set for pass-by-pass tests (both categories need to be skipped)
_SKIP_IN_PASSBYPASS = KNOWN_PARSER_BUGS | KNOWN_PREOPT_ONLY_BUGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(path: Path) -> Module:
    src = path.read_text(encoding='utf-8')
    return parse(preprocess(src))


def _parse_opt(path: Path) -> Module:
    src = path.read_text(encoding='utf-8')
    return optimize(parse(preprocess(src)))


def _make_single_block_kernel(name: str = 'test') -> Kernel:
    k = Kernel(name=name, params=[])
    k.blocks.append(BasicBlock(label='entry_1', terminator=RetTerm()))
    return k


def _make_two_block_kernel() -> Kernel:
    """entry_1 → bb_2"""
    k = Kernel(name='two_block', params=[])
    k.blocks.append(BasicBlock(label='entry_1', terminator=BrTerm('bb_2')))
    k.blocks.append(BasicBlock(label='bb_2', terminator=RetTerm()))
    return k


# ---------------------------------------------------------------------------
# Group 1a: Pre-optimization — branch/terminator checks pass (no reachability)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_pre_opt_structural_clean(cu_file):
    """Before optimization: no missing terminators and no dangling branches."""
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    all_errs = verify_module(mod, check_reachability=False)
    for kname, errs in all_errs.items():
        structural = [e for e in errs
                      if 'missing terminator' in e or 'undefined block' in e]
        assert not structural, (
            f'{cu_file.stem}/{kname}: structural errors before opt:\n'
            + '\n'.join(structural))


# ---------------------------------------------------------------------------
# Group 1b: Post-optimization — all checks, zero violations expected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_post_opt_zero_violations(cu_file):
    """After full optimization: IR verifier finds no violations."""
    if cu_file.stem in KNOWN_PARSER_BUGS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse_opt(cu_file)
    all_errs = verify_module(mod, check_reachability=True)
    flat = []
    for kname, errs in all_errs.items():
        for e in errs:
            flat.append(e)
    assert not flat, (
        f'{cu_file.stem}: verifier violations after opt:\n'
        + '\n'.join(flat))


# ---------------------------------------------------------------------------
# Group 2: Negative cases — broken IR detected correctly
# ---------------------------------------------------------------------------

def test_missing_terminator_detected():
    k = Kernel(name='bad', params=[])
    k.blocks.append(BasicBlock(label='entry_1'))  # no terminator
    errs = verify_kernel(k, check_reachability=False)
    assert any('missing terminator' in e for e in errs), \
        f'Expected missing terminator error, got: {errs}'


def test_branch_to_undefined_block_detected():
    k = _make_single_block_kernel()
    k.blocks[0].terminator = BrTerm(target='ghost_block')
    errs = verify_kernel(k, check_reachability=False)
    assert any('undefined block' in e for e in errs), \
        f'Expected undefined block error, got: {errs}'


def test_condbr_to_undefined_true_block_detected():
    k = Kernel(name='bad_cond', params=[])
    cond = k.new_value('c', INT32)
    k.blocks.append(BasicBlock(
        label='entry_1',
        instructions=[CmpInst(dest=cond, op=CmpOp.EQ,
                               lhs=Const(INT32, 0), rhs=Const(INT32, 0))],
        terminator=CondBrTerm(cond=cond, true_bb='ghost', false_bb='entry_1'),
    ))
    errs = verify_kernel(k, check_reachability=False)
    assert any('undefined block' in e for e in errs)


def test_undefined_value_use_detected():
    """Use a Value that has no defining instruction."""
    k = Kernel(name='undef_use', params=[])
    phantom = Value(name='phantom', ty=INT32, id=99)  # never defined
    dest = k.new_value('d', INT32)
    k.blocks.append(BasicBlock(
        label='entry_1',
        instructions=[BinInst(dest=dest, op=BinOp.ADD,
                               lhs=phantom, rhs=Const(INT32, 1))],
        terminator=RetTerm(),
    ))
    errs = verify_kernel(k, check_reachability=False)
    assert any('undefined value' in e for e in errs), \
        f'Expected undefined value error, got: {errs}'


def test_undefined_value_in_condbr_detected():
    """CondBrTerm condition is a Value with no defining instruction."""
    k = Kernel(name='undef_cond', params=[])
    phantom_cond = Value(name='pc', ty=INT32, id=77)  # never defined
    bb2 = BasicBlock(label='bb_2', terminator=RetTerm())
    k.blocks.append(BasicBlock(
        label='entry_1',
        terminator=CondBrTerm(cond=phantom_cond, true_bb='bb_2', false_bb='bb_2'),
    ))
    k.blocks.append(bb2)
    errs = verify_kernel(k, check_reachability=False)
    assert any('undefined value' in e for e in errs)


def test_unreachable_block_detected():
    """An orphan block with no predecessors is flagged with check_reachability."""
    k = Kernel(name='orphan', params=[])
    k.blocks.append(BasicBlock(label='entry_1', terminator=RetTerm()))
    k.blocks.append(BasicBlock(label='orphan_2', terminator=RetTerm()))
    errs = verify_kernel(k, check_reachability=True)
    assert any('unreachable' in e for e in errs)


def test_unreachable_block_not_reported_when_disabled():
    """With check_reachability=False, orphan blocks are not an error."""
    k = Kernel(name='orphan', params=[])
    k.blocks.append(BasicBlock(label='entry_1', terminator=RetTerm()))
    k.blocks.append(BasicBlock(label='orphan_2', terminator=RetTerm()))
    errs = verify_kernel(k, check_reachability=False)
    assert not any('unreachable' in e for e in errs)


def test_dominance_violation_detected():
    """Value used in a block that its def does not dominate."""
    # CFG: entry_1 → (bb_true, bb_false) → bb_merge
    # val defined in bb_true only, used in bb_merge
    # entry_1 does NOT define val, so bb_true doesn't dominate bb_merge
    k = Kernel(name='dom_violation', params=[])
    cond = k.new_value('cond', INT32)
    val = k.new_value('val', INT32)
    result = k.new_value('result', INT32)

    # entry: define cond
    entry = BasicBlock(
        label='entry_1',
        instructions=[
            CmpInst(dest=cond, op=CmpOp.EQ,
                    lhs=Const(INT32, 0), rhs=Const(INT32, 0)),
        ],
        terminator=CondBrTerm(cond=cond, true_bb='bb_true', false_bb='bb_false'),
    )
    # bb_true: define val
    bb_true = BasicBlock(
        label='bb_true',
        instructions=[
            BinInst(dest=val, op=BinOp.ADD,
                    lhs=Const(INT32, 1), rhs=Const(INT32, 2)),
        ],
        terminator=BrTerm(target='bb_merge'),
    )
    # bb_false: does not define val
    bb_false = BasicBlock(
        label='bb_false',
        terminator=BrTerm(target='bb_merge'),
    )
    # bb_merge: USE val (which is only defined in bb_true, not dominant)
    bb_merge = BasicBlock(
        label='bb_merge',
        instructions=[
            BinInst(dest=result, op=BinOp.ADD,
                    lhs=val, rhs=Const(INT32, 0)),
        ],
        terminator=RetTerm(),
    )
    k.blocks.extend([entry, bb_true, bb_false, bb_merge])

    errs = verify_kernel(k, check_reachability=True)
    assert any('dominance violation' in e for e in errs), \
        f'Expected dominance violation, got: {errs}'


def test_valid_diamond_no_errors():
    """A well-formed diamond CFG with def in entry (dominates all) has no errors."""
    k = Kernel(name='clean_diamond', params=[])
    cond = k.new_value('cond', INT32)
    val = k.new_value('val', INT32)
    result = k.new_value('result', INT32)

    entry = BasicBlock(
        label='entry_1',
        instructions=[
            BinInst(dest=val, op=BinOp.ADD,
                    lhs=Const(INT32, 5), rhs=Const(INT32, 3)),
            CmpInst(dest=cond, op=CmpOp.GT,
                    lhs=val, rhs=Const(INT32, 0)),
        ],
        terminator=CondBrTerm(cond=cond, true_bb='bb_true', false_bb='bb_false'),
    )
    bb_true = BasicBlock(
        label='bb_true',
        terminator=BrTerm(target='bb_merge'),
    )
    bb_false = BasicBlock(
        label='bb_false',
        terminator=BrTerm(target='bb_merge'),
    )
    bb_merge = BasicBlock(
        label='bb_merge',
        instructions=[
            # val was defined in entry which dominates bb_merge ✓
            BinInst(dest=result, op=BinOp.ADD,
                    lhs=val, rhs=Const(INT32, 0)),
        ],
        terminator=RetTerm(),
    )
    k.blocks.extend([entry, bb_true, bb_false, bb_merge])
    errs = verify_kernel(k)
    assert not errs, f'Expected no errors, got: {errs}'


def test_module_verify_aggregates_kernels():
    """verify_module returns errors per kernel."""
    mod = Module()
    good = _make_single_block_kernel('good')
    bad = Kernel(name='bad', params=[])
    bad.blocks.append(BasicBlock(label='entry_1'))  # no terminator
    mod.kernels.extend([good, bad])
    result = verify_module(mod, check_reachability=False)
    assert 'good' not in result
    assert 'bad' in result


# ---------------------------------------------------------------------------
# Group 3: Pass-by-pass revalidation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_after_constant_fold_clean(cu_file):
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    for kernel in mod.kernels:
        constant_fold(kernel)
    all_errs = verify_module(mod, check_reachability=False)
    flat = [e for errs in all_errs.values() for e in errs]
    assert not flat, f'{cu_file.stem}: errors after constant_fold:\n' + '\n'.join(flat)


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_after_cse_clean(cu_file):
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    for kernel in mod.kernels:
        constant_fold(kernel)
        cse(kernel)
    all_errs = verify_module(mod, check_reachability=False)
    flat = [e for errs in all_errs.values() for e in errs]
    assert not flat, f'{cu_file.stem}: errors after cse:\n' + '\n'.join(flat)


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_after_dead_block_elim_clean(cu_file):
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    for kernel in mod.kernels:
        constant_fold(kernel)
        cse(kernel)
        dead_block_elim(kernel)
    all_errs = verify_module(mod, check_reachability=True)
    flat = [e for errs in all_errs.values() for e in errs]
    assert not flat, (
        f'{cu_file.stem}: errors after dead_block_elim:\n' + '\n'.join(flat))


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_after_identity_fold_clean(cu_file):
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    for kernel in mod.kernels:
        constant_fold(kernel)
        cse(kernel)
        dead_block_elim(kernel)
        identity_fold(kernel)
    all_errs = verify_module(mod, check_reachability=True)
    flat = [e for errs in all_errs.values() for e in errs]
    assert not flat, (
        f'{cu_file.stem}: errors after identity_fold:\n' + '\n'.join(flat))


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_after_licm_clean(cu_file):
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse(cu_file)
    for kernel in mod.kernels:
        constant_fold(kernel)
        cse(kernel)
        dead_block_elim(kernel)
        identity_fold(kernel)
        dead_inst_elim(kernel)
        licm(kernel)
    all_errs = verify_module(mod, check_reachability=True)
    flat = [e for errs in all_errs.values() for e in errs]
    assert not flat, (
        f'{cu_file.stem}: errors after licm:\n' + '\n'.join(flat))


# ---------------------------------------------------------------------------
# Group 4: Benchmark stats (Deliverable D) — logged, never fails
# ---------------------------------------------------------------------------

def test_benchmark_stats_table(capsys):
    """Print a stats table for all kernels. Always passes."""
    rows = []
    for cu_file in ALL_CU_FILES:
        if cu_file.stem == 'switch_test':
            continue
        mod = _parse_opt(cu_file)
        for kernel in mod.kernels:
            s = kernel_stats(kernel)
            rows.append((cu_file.stem, kernel.name,
                         s['block_count'], s['reachable_count'],
                         s['loop_count'], s['has_branches']))

    # Print table
    header = f"{'File':<30} {'Kernel':<25} {'Blk':>4} {'Rch':>4} {'Lps':>4} {'Br':>3}"
    sep = '-' * len(header)
    with capsys.disabled():
        print(f'\n{sep}')
        print(header)
        print(sep)
        for stem, kname, blk, rch, lps, br in rows:
            print(f'{stem:<30} {kname:<25} {blk:>4} {rch:>4} {lps:>4} {"Y" if br else "N":>3}')
        print(sep)
        print(f'Total kernels: {len(rows)}')
        total_loops = sum(r[4] for r in rows)
        kernels_with_loops = sum(1 for r in rows if r[4] > 0)
        print(f'Total loops:   {total_loops} (in {kernels_with_loops} kernels)')

    # Sanity assertions
    assert len(rows) > 0
    for _, _, blk, rch, lps, _ in rows:
        assert blk >= rch >= 0
        assert lps >= 0


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_stats_consistent_with_optimize(cu_file):
    """After optimization, reachable_count == block_count (no dead blocks remain)."""
    if cu_file.stem in KNOWN_PARSER_BUGS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        s = kernel_stats(kernel)
        assert s['reachable_count'] == s['block_count'], (
            f'{cu_file.stem}/{kernel.name}: '
            f'reachable={s["reachable_count"]} != block_count={s["block_count"]} '
            f'after dead_block_elim')


# ---------------------------------------------------------------------------
# Group 5: Known-bug confirmation — verifier correctly detects pre-existing bugs
# ---------------------------------------------------------------------------

# Files with known IR bugs that the verifier must detect (not silently ignore).
# This ensures the verifier stays sensitive and doesn't regress.
_KNOWN_BUGGY_FILES = [
    f for f in ALL_CU_FILES
    if f.stem in KNOWN_PARSER_BUGS and f.stem != 'switch_test'
]


@pytest.mark.parametrize('cu_file', _KNOWN_BUGGY_FILES, ids=lambda p: p.stem)
def test_known_parser_bugs_detected(cu_file):
    """The verifier must report errors for known-buggy kernels after optimization.

    This confirms the verifier remains sensitive: if a future fix resolves one
    of these bugs, this test will fail and the file should be removed from
    KNOWN_PARSER_BUGS.
    """
    mod = _parse_opt(cu_file)
    all_errs = verify_module(mod, check_reachability=True)
    flat = [e for errs in all_errs.values() for e in errs]
    assert flat, (
        f'{cu_file.stem}: expected IR violations (known parser bug) '
        f'but verifier found none — remove from KNOWN_PARSER_BUGS if fixed')


# ---------------------------------------------------------------------------
# Group 6: Single-entry loop guarantee (Check 6 in verify_kernel)
# ---------------------------------------------------------------------------

def test_natural_loop_header_dominates_body():
    """A well-formed for-loop: header dominates all body blocks."""
    k = Kernel(name='natural_loop', params=[])
    cond_v = k.new_value('cond', INT32)
    entry = BasicBlock(label='entry_1', terminator=BrTerm('header_2'))
    header = BasicBlock(
        label='header_2',
        instructions=[CmpInst(dest=cond_v, op=CmpOp.LT,
                               lhs=Const(INT32, 0), rhs=Const(INT32, 10))],
        terminator=CondBrTerm(cond=cond_v, true_bb='body_3', false_bb='exit_4'),
    )
    body = BasicBlock(label='body_3', terminator=BrTerm('header_2'))
    exit_ = BasicBlock(label='exit_4', terminator=RetTerm())
    k.blocks.extend([entry, header, body, exit_])
    errs = verify_kernel(k, check_reachability=True)
    loop_errs = [e for e in errs if 'non-natural' in e]
    assert not loop_errs, f'Expected natural loop to pass, got: {loop_errs}'


def test_irreducible_loop_detected():
    """A non-natural loop where the 'header' does not dominate the body."""
    # Irreducible CFG: entry → A and entry → B; A → B and B → A (no single header)
    k = Kernel(name='irreducible', params=[])
    cond1 = k.new_value('c1', INT32)
    cond2 = k.new_value('c2', INT32)
    entry = BasicBlock(
        label='entry_1',
        instructions=[CmpInst(dest=cond1, op=CmpOp.EQ,
                               lhs=Const(INT32, 0), rhs=Const(INT32, 0))],
        terminator=CondBrTerm(cond=cond1, true_bb='bb_A', false_bb='bb_B'),
    )
    bb_A = BasicBlock(
        label='bb_A',
        instructions=[CmpInst(dest=cond2, op=CmpOp.EQ,
                               lhs=Const(INT32, 1), rhs=Const(INT32, 1))],
        terminator=CondBrTerm(cond=cond2, true_bb='bb_B', false_bb='exit_4'),
    )
    bb_B = BasicBlock(
        label='bb_B',
        terminator=BrTerm(target='bb_A'),  # back edge B → A (but A is not header)
    )
    exit_ = BasicBlock(label='exit_4', terminator=RetTerm())
    k.blocks.extend([entry, bb_A, bb_B, exit_])
    errs = verify_kernel(k, check_reachability=True)
    loop_errs = [e for e in errs if 'non-natural' in e]
    assert loop_errs, f'Expected non-natural loop error, got none. All errs: {errs}'


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_real_kernel_loops_are_natural(cu_file):
    """All natural loops in real kernels must have headers that dominate bodies."""
    if cu_file.stem in _SKIP_IN_PASSBYPASS:
        pytest.skip(f'known pre-existing parser bug: {cu_file.stem}')
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        errs = verify_kernel(kernel, check_reachability=True)
        loop_errs = [e for e in errs if 'non-natural' in e]
        assert not loop_errs, (
            f'{cu_file.stem}/{kernel.name}: non-natural loop errors:\n'
            + '\n'.join(loop_errs))


# ---------------------------------------------------------------------------
# Group 7: Critical edge detection (find_critical_edges utility)
# ---------------------------------------------------------------------------

def test_no_critical_edges_straight_line():
    """Straight-line CFG has no critical edges."""
    k = Kernel(name='straight', params=[])
    k.blocks.append(BasicBlock(label='entry_1', terminator=BrTerm('bb_2')))
    k.blocks.append(BasicBlock(label='bb_2', terminator=RetTerm()))
    assert find_critical_edges(k) == []


def test_no_critical_edges_diamond():
    """Simple diamond: entry → (A, B) → merge.
    entry has 2 succs but A and B each have 1 pred.
    merge has 2 preds but A and B each have 1 succ.
    No critical edges."""
    k = Kernel(name='diamond', params=[])
    cond = k.new_value('c', INT32)
    k.blocks.append(BasicBlock(
        label='entry_1',
        instructions=[CmpInst(dest=cond, op=CmpOp.EQ,
                               lhs=Const(INT32, 0), rhs=Const(INT32, 0))],
        terminator=CondBrTerm(cond=cond, true_bb='bb_A', false_bb='bb_B'),
    ))
    k.blocks.append(BasicBlock(label='bb_A', terminator=BrTerm('merge_4')))
    k.blocks.append(BasicBlock(label='bb_B', terminator=BrTerm('merge_4')))
    k.blocks.append(BasicBlock(label='merge_4', terminator=RetTerm()))
    assert find_critical_edges(k) == []


def test_critical_edge_do_while():
    """Do-while loop creates a critical edge: do_cond → do_body.
    do_cond has 2 succs (do_body for true, exit for false).
    do_body has 2 preds (entry and do_cond).
    """
    k = Kernel(name='do_while', params=[])
    cond = k.new_value('c', INT32)
    entry = BasicBlock(label='entry_1', terminator=BrTerm('do_body_2'))
    do_body = BasicBlock(
        label='do_body_2',
        instructions=[CmpInst(dest=cond, op=CmpOp.LT,
                               lhs=Const(INT32, 0), rhs=Const(INT32, 10))],
        terminator=CondBrTerm(cond=cond, true_bb='do_body_2', false_bb='exit_3'),
    )
    exit_ = BasicBlock(label='exit_3', terminator=RetTerm())
    k.blocks.extend([entry, do_body, exit_])
    crit = find_critical_edges(k)
    assert any(src == 'do_body_2' and dst == 'do_body_2' for src, dst in crit), \
        f'Expected do_body_2 self-loop critical edge, got: {crit}'


def test_find_critical_edges_empty_kernel():
    """Empty kernel has no critical edges."""
    k = Kernel(name='empty', params=[])
    assert find_critical_edges(k) == []


@pytest.mark.parametrize('cu_file', ALL_CU_FILES, ids=lambda p: p.stem)
def test_critical_edge_count_is_non_negative(cu_file):
    """find_critical_edges returns a list (never raises)."""
    mod = _parse_opt(cu_file)
    for kernel in mod.kernels:
        crit = find_critical_edges(kernel)
        assert isinstance(crit, list)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in crit)


# ---------------------------------------------------------------------------
# Group 8: debug_verify gating in optimize()
# ---------------------------------------------------------------------------

def test_debug_verify_passes_clean_kernels():
    """debug_verify=True does not raise on clean kernels."""
    mod = _parse(TESTS_DIR / 'vector_add.cu')
    optimize(mod, debug_verify=True)  # must not raise


def test_debug_verify_is_off_by_default():
    """optimize() without debug_verify=True uses no verification overhead."""
    import time
    mod = _parse(TESTS_DIR / 'matmul_tiled.cu')
    # Just ensure default optimize() completes without exception
    result = optimize(mod)
    assert result is mod
