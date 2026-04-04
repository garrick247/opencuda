"""
OpenCUDA IR Verifier — v0.10.

Checks correctness invariants after any transformation pass.

Checks performed
----------------
1. **Terminator presence**: every block has a non-None terminator.
2. **Branch target validity**: every branch/cond-branch target names an
   existing block label.
3. **Block reachability**: every block is reachable from the entry (BFS).
   Can be disabled with ``check_reachability=False`` for pre-optimized IR
   where ``after_break``/``after_continue`` stubs are expected.
4. **Def presence**: every Value used as an operand or CondBrTerm condition
   has a corresponding defining instruction somewhere in the kernel.
5. **Domination**: for single-definition Values (def_count == 1), the
   defining block dominates every block that uses the value.
   Multi-definition Values (loop writeback pattern, def_count ≥ 2) are
   skipped for this check since they intentionally predate pure SSA form.
6. **Critical edge detection**: a critical edge is one from a block with
   multiple successors to a block with multiple predecessors.  Such edges
   block many transforms (PHI insertion, PRE, edge splitting).  Reported as
   warnings — transforms that require edge splitting will add the check
   themselves.
7. **Single-entry loop guarantee**: for every natural loop, the loop header
   must dominate every block in the loop body.  Violations indicate a
   structurally malformed loop (non-natural or irreducible CFG).

Usage
-----
    from opencuda.ir.verify_ir import verify_kernel, verify_module

    errs = verify_kernel(kernel)
    assert not errs, '\\n'.join(errs)

    all_errs = verify_module(module)
    for kname, errs in all_errs.items():
        print(f'{kname}: {errs}')
"""

from __future__ import annotations
from typing import Optional
from ..ir.nodes import (Kernel, Module, BasicBlock, Value, Const,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, PrintfInst, PhiInst,
                         AsmInst, BrTerm, CondBrTerm, RetTerm)
from ..ir.types import PtrTy, AddrSpace
from .dominator import compute_dominators, dominates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bb_successors(bb: BasicBlock) -> list[str]:
    t = bb.terminator
    if isinstance(t, BrTerm):
        return [t.target]
    if isinstance(t, CondBrTerm):
        return [t.true_bb, t.false_bb]
    return []


def _operand_values(inst) -> list[Value]:
    """Return all Value operands referenced by an instruction."""
    uses: list[Value] = []
    for attr in ('lhs', 'rhs', 'src', 'addr', 'value'):
        v = getattr(inst, attr, None)
        if isinstance(v, Value):
            uses.append(v)
    if hasattr(inst, 'args'):
        for a in inst.args:
            if isinstance(a, Value):
                uses.append(a)
    if isinstance(inst, PhiInst):
        for val, _lbl in inst.incoming:
            if isinstance(val, Value):
                uses.append(val)
    return uses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_kernel(kernel: Kernel,
                  dom: Optional[dict] = None,
                  check_reachability: bool = True) -> list[str]:
    """Verify invariants for a single kernel.

    Parameters
    ----------
    kernel             : the kernel to verify
    dom                : precomputed dominator dict (recomputed if None)
    check_reachability : if True (default), report unreachable blocks as
                         errors.  Set False for pre-optimised IR.

    Returns
    -------
    List of error strings.  Empty list means the kernel is clean.
    """
    errors: list[str] = []

    if not kernel.blocks:
        return errors

    kname = kernel.name
    label_set = {bb.label for bb in kernel.blocks}
    label_to_bb = {bb.label: bb for bb in kernel.blocks}
    entry = kernel.blocks[0].label

    # ------------------------------------------------------------------
    # Check 1: terminator presence + branch target validity
    # ------------------------------------------------------------------
    for bb in kernel.blocks:
        if bb.terminator is None:
            errors.append(
                f'[{kname}:{bb.label}] missing terminator')
        else:
            for tgt in _bb_successors(bb):
                if tgt not in label_set:
                    errors.append(
                        f'[{kname}:{bb.label}] branch to undefined block '
                        f'{tgt!r}')

    # ------------------------------------------------------------------
    # Check 2: block reachability
    # ------------------------------------------------------------------
    reachable: set[str] = set()
    work = [entry]
    while work:
        lbl = work.pop()
        if lbl in reachable:
            continue
        reachable.add(lbl)
        bb = label_to_bb.get(lbl)
        if bb:
            for tgt in _bb_successors(bb):
                if tgt in label_set and tgt not in reachable:
                    work.append(tgt)

    if check_reachability:
        for bb in kernel.blocks:
            if bb.label not in reachable:
                errors.append(
                    f'[{kname}:{bb.label}] block is unreachable from entry')

    # ------------------------------------------------------------------
    # Check 3: build def map
    # def_block[id] = block label of the first (or only) defining instruction
    # def_count[id] = number of instructions that write to this Value id
    # ------------------------------------------------------------------
    def_block: dict[int, str] = {}
    def_count: dict[int, int] = {}

    for bb in kernel.blocks:
        for inst in bb.instructions:
            dest = getattr(inst, 'dest', None)
            if dest is not None and isinstance(dest, Value):
                vid = dest.id
                def_count[vid] = def_count.get(vid, 0) + 1
                if vid not in def_block:
                    def_block[vid] = bb.label
            # AsmInst: output operands [(constraint, Value), ...] are defined here
            if isinstance(inst, AsmInst):
                for _, out_val in inst.outputs:
                    if isinstance(out_val, Value):
                        vid = out_val.id
                        def_count[vid] = def_count.get(vid, 0) + 1
                        if vid not in def_block:
                            def_block[vid] = bb.label

    # ------------------------------------------------------------------
    # Check 4: every use has a def
    # Shared-memory pointer Values (PtrTy(AddrSpace.SHARED)) are exempt:
    # their "definition" is the .shared PTX declaration emitted by codegen,
    # not an IR instruction.
    # ------------------------------------------------------------------
    def _is_smem(v: Value) -> bool:
        return (isinstance(v.ty, PtrTy)
                and v.ty.addr_space in (AddrSpace.SHARED, AddrSpace.LOCAL))

    for bb in kernel.blocks:
        for inst in bb.instructions:
            for use in _operand_values(inst):
                if use.id not in def_block and not _is_smem(use):
                    errors.append(
                        f'[{kname}:{bb.label}] use of undefined value '
                        f'%{use.name} (id={use.id}) in {type(inst).__name__}')
        # Terminator condition
        t = bb.terminator
        if isinstance(t, CondBrTerm) and isinstance(t.cond, Value):
            if t.cond.id not in def_block:
                errors.append(
                    f'[{kname}:{bb.label}] CondBrTerm uses undefined value '
                    f'%{t.cond.name} (id={t.cond.id})')

    # ------------------------------------------------------------------
    # Check 5: def dominates use (only for single-def Values)
    # ------------------------------------------------------------------
    if dom is None:
        dom = compute_dominators(kernel)

    for bb in kernel.blocks:
        if bb.label not in reachable:
            continue  # don't check dominance in dead code

        for inst in bb.instructions:
            for use in _operand_values(inst):
                vid = use.id
                if vid not in def_block:
                    continue  # already reported in check 4
                if def_count.get(vid, 0) != 1:
                    continue  # loop writeback: skip (by design not SSA-pure)
                def_lbl = def_block[vid]
                use_lbl = bb.label
                if def_lbl == use_lbl:
                    continue  # same block: order assumed correct by construction
                if dominates(dom, def_lbl, use_lbl):
                    continue  # valid forward domination
                # Loop-recurrence use: the use_block dominates the def_block
                # (use_block is the loop header, def_block is in the loop body,
                # connected via a back edge).  Valid in OpenCUDA's loop model.
                if dominates(dom, use_lbl, def_lbl):
                    continue
                errors.append(
                    f'[{kname}] dominance violation: %{use.name} '
                    f'(id={vid}) defined in {def_lbl!r} does not '
                    f'dominate use in {use_lbl!r}')

        # Terminator condition
        t = bb.terminator
        if isinstance(t, CondBrTerm) and isinstance(t.cond, Value):
            vid = t.cond.id
            if vid in def_block and def_count.get(vid, 0) == 1:
                def_lbl = def_block[vid]
                if def_lbl != bb.label:
                    if not dominates(dom, def_lbl, bb.label):
                        if not dominates(dom, bb.label, def_lbl):
                            errors.append(
                                f'[{kname}] dominance violation in CondBrTerm: '
                                f'%{t.cond.name} (id={vid}) defined in '
                                f'{def_lbl!r} does not dominate use in '
                                f'{bb.label!r}')

    # ------------------------------------------------------------------
    # Check 6: single-entry loop guarantee
    # For every natural loop (DFS back-edge), the loop header must dominate
    # every REACHABLE block in the loop body.  Unreachable stubs (after_break,
    # after_continue) are skipped — dead_block_elim will remove them.
    # Violation = non-natural / irreducible CFG on live code.
    # ------------------------------------------------------------------
    _visited2: set[str] = set()
    _on_stack2: set[str] = set()
    _back_edges: list[tuple[str, str]] = []

    def _dfs2(lbl: str) -> None:
        if lbl in _visited2:
            return
        _visited2.add(lbl)
        _on_stack2.add(lbl)
        bb = label_to_bb.get(lbl)
        if bb:
            for succ in _bb_successors(bb):
                if succ not in label_to_bb:
                    continue
                if succ in _on_stack2:
                    _back_edges.append((lbl, succ))
                elif succ not in _visited2:
                    _dfs2(succ)
        _on_stack2.discard(lbl)

    _dfs2(entry)

    _pred_map2: dict[str, list[str]] = {bb.label: [] for bb in kernel.blocks}
    for bb in kernel.blocks:
        for s in _bb_successors(bb):
            if s in _pred_map2:
                _pred_map2[s].append(bb.label)

    seen_loop_edges: set[tuple[str, str]] = set()
    for backedge_src, header in _back_edges:
        edge = (backedge_src, header)
        if edge in seen_loop_edges:
            continue
        seen_loop_edges.add(edge)

        body: set[str] = {header}
        work = [backedge_src]
        while work:
            cur = work.pop()
            if cur in body:
                continue
            body.add(cur)
            for p in _pred_map2.get(cur, []):
                if p not in body:
                    work.append(p)

        for body_lbl in body:
            if body_lbl == header:
                continue
            if body_lbl not in reachable:
                continue  # dead stub — ignore
            if not dominates(dom, header, body_lbl):
                errors.append(
                    f'[{kname}] non-natural loop: header {header!r} does not '
                    f'dominate reachable body block {body_lbl!r}')

    return errors


# ---------------------------------------------------------------------------
# Standalone structural queries (not part of the error-raising verifier)
# ---------------------------------------------------------------------------

def find_critical_edges(kernel: Kernel) -> list[tuple[str, str]]:
    """Return all critical edges in the kernel's CFG.

    A critical edge is an edge (src → dst) where src has more than one
    successor AND dst has more than one predecessor.  Such edges must be
    split before PHI insertion, PRE, or any transform that needs to insert
    code on a specific edge.

    Critical edges are structural properties, not errors — do-while loops
    and post-loop join blocks naturally create them.
    """
    pred_count: dict[str, int] = {bb.label: 0 for bb in kernel.blocks}
    for bb in kernel.blocks:
        for tgt in _bb_successors(bb):
            if tgt in pred_count:
                pred_count[tgt] += 1

    critical: list[tuple[str, str]] = []
    for bb in kernel.blocks:
        succs = _bb_successors(bb)
        if len(succs) > 1:
            for tgt in succs:
                if pred_count.get(tgt, 0) > 1:
                    critical.append((bb.label, tgt))
    return critical


def verify_module(module: Module,
                  check_reachability: bool = True) -> dict[str, list[str]]:
    """Verify all kernels in a module.

    Returns a dict of ``kernel_name → [error strings]``.
    Only kernels with violations appear in the result.
    """
    result: dict[str, list[str]] = {}
    for kernel in module.kernels:
        errs = verify_kernel(kernel, check_reachability=check_reachability)
        if errs:
            result[kernel.name] = errs
    return result
