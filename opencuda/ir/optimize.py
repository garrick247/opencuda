"""
OpenCUDA IR optimization passes.

Pass 1: Constant folding — evaluate Const op Const at compile time.
Pass 2: CSE — reuse identical computations within a basic block.
        v0.7: extended with commutative normalization (ADD/MUL/AND/OR/XOR,
        EQ/NE), reversible CmpOp normalization (LT↔GT, LE↔GE), and
        full CmpInst deduplication.
Pass 3: Dead block elimination — remove unreachable basic blocks.
Pass 4: Identity fold / copy propagation — eliminate add D, S, 0 for
        single-definition Values.
Pass 5: Dead instruction elimination — remove BinInst/CmpInst/CvtInst
        whose result is never consumed.
Pass 6: Post-CSE cleanup — second CSE + identity_fold + dead_inst_elim
        round to catch expressions exposed by pass 3–5.

SAFETY RULE for passes 1–2: Replacements never cross basic block boundaries.
This prevents the loop writeback bug where a variable initialized
in the entry block (float sum = 0) gets replaced by Const(0) in
the loop body, causing the loop condition to never see updates.

SAFETY RULE for pass 4 (identity_fold): only propagates single-definition
Values (def_count == 1). Loop-writeback Values are redefined each iteration
and therefore have def_count >= 2 — they are never touched.
"""

from __future__ import annotations
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, PrintfInst, BinOp, CmpOp,
                         BrTerm, CondBrTerm, RetTerm)
from ..ir.types import INT32, UINT32, FLOAT, ScalarTy


def _const_val(op: Operand):
    if isinstance(op, Const):
        return op.value
    return None


def _fold_bin(op: BinOp, a, b, is_float: bool):
    if a is None or b is None:
        return None
    try:
        if is_float:
            a, b = float(a), float(b)
        else:
            a, b = int(a), int(b)
        if op == BinOp.ADD: return a + b
        if op == BinOp.SUB: return a - b
        if op == BinOp.MUL: return a * b
        if op == BinOp.DIV and b != 0: return a // b if not is_float else a / b
        if op == BinOp.MOD and b != 0: return a % b
        if op == BinOp.AND: return int(a) & int(b)
        if op == BinOp.OR:  return int(a) | int(b)
        if op == BinOp.XOR: return int(a) ^ int(b)
        if op == BinOp.SHL: return int(a) << int(b)
        if op == BinOp.SHR: return int(a) >> int(b)
    except:
        pass
    return None


def constant_fold(kernel: Kernel) -> int:
    """
    Fold Const op Const into a single Const.
    Also folds safe identity ops: x * 0 → 0.

    SAFETY: Replacements are LOCAL to each basic block. A fold in the
    entry block does NOT propagate to the loop body. This prevents the
    loop writeback bug.

    SAFETY: We never put Const results into the cross-instruction
    replacement map. A folded instruction is simply removed; its
    Value ceases to exist. Only Value→Value replacements (identity
    folds where dest and replacement are both registers) propagate.
    """
    folded = 0

    for bb in kernel.blocks:
        # Per-block replacement map — does NOT leak to other blocks
        replacements: dict[int, Operand] = {}

        def _resolve(op: Operand) -> Operand:
            if isinstance(op, Value) and op.id in replacements:
                r = replacements[op.id]
                # Follow chains but only for Value→Value
                while isinstance(r, Value) and r.id in replacements:
                    r = replacements[r.id]
                return r
            return op

        new_insts = []
        for inst in bb.instructions:
            if isinstance(inst, BinInst):
                lhs = _resolve(inst.lhs)
                rhs = _resolve(inst.rhs)
                inst.lhs = lhs
                inst.rhs = rhs

                is_float = isinstance(inst.dest.ty, ScalarTy) and inst.dest.ty.is_float
                lv = _const_val(lhs)
                rv = _const_val(rhs)

                # Full constant fold (both operands are Const)
                result = _fold_bin(inst.op, lv, rv, is_float)
                if result is not None:
                    # DON'T put in replacements — the dest Value might be
                    # read in another block. Just emit a materialization.
                    # Actually: we CAN fold if we emit the constant inline
                    # wherever dest is used. But that's complex. Instead:
                    # emit a simpler instruction (mov-like: add dest, const, 0)
                    # with the folded value.
                    if is_float:
                        inst.lhs = Const(inst.dest.ty, result)
                        inst.rhs = Const(inst.dest.ty, 0.0)
                        inst.op = BinOp.ADD
                    else:
                        inst.lhs = Const(inst.dest.ty, int(result))
                        inst.rhs = Const(inst.dest.ty, 0)
                        inst.op = BinOp.ADD
                    folded += 1
                    # Keep the instruction (it's now simpler but still writes dest)

                # Strength reduction: mul by power of 2 → shift left (integers only)
                elif not is_float and inst.op == BinOp.MUL and rv is not None and isinstance(rv, int) and rv > 0 and (rv & (rv-1)) == 0:
                    shift = rv.bit_length() - 1
                    inst.op = BinOp.SHL
                    inst.rhs = Const(inst.dest.ty, shift)
                    folded += 1
                elif not is_float and inst.op == BinOp.MUL and lv is not None and isinstance(lv, int) and lv > 0 and (lv & (lv-1)) == 0:
                    shift = lv.bit_length() - 1
                    inst.op = BinOp.SHL
                    inst.lhs = inst.rhs
                    inst.rhs = Const(inst.dest.ty, shift)
                    folded += 1

                # Safe identity fold: x * 0 → replace instruction with "add dest, 0, 0"
                elif inst.op == BinOp.MUL and (rv == 0 or lv == 0):
                    inst.lhs = Const(inst.dest.ty, 0)
                    inst.rhs = Const(inst.dest.ty, 0)
                    inst.op = BinOp.ADD
                    folded += 1

            elif isinstance(inst, CmpInst):
                inst.lhs = _resolve(inst.lhs)
                inst.rhs = _resolve(inst.rhs)
            elif isinstance(inst, LoadInst):
                inst.addr = _resolve(inst.addr)
            elif isinstance(inst, StoreInst):
                inst.addr = _resolve(inst.addr)
                inst.value = _resolve(inst.value)

            new_insts.append(inst)
        bb.instructions = new_insts

    return folded


_COMMUTATIVE_BINOPS = frozenset({BinOp.ADD, BinOp.MUL, BinOp.AND, BinOp.OR, BinOp.XOR})

# CmpOp pairs where swapping operands yields an equivalent comparison
_CMP_SWAP = {CmpOp.LT: CmpOp.GT, CmpOp.GT: CmpOp.LT,
             CmpOp.LE: CmpOp.GE, CmpOp.GE: CmpOp.LE}
_CMP_COMMUTATIVE = frozenset({CmpOp.EQ, CmpOp.NE})


def cse(kernel: Kernel) -> int:
    """
    Common Subexpression Elimination (local, per basic block).

    Eliminates duplicate BinInst, CvtInst, and CmpInst within a block.
    Commutative BinInst (ADD/MUL/AND/OR/XOR) and symmetric CmpInst
    (EQ/NE) are normalized so that a+b and b+a share the same key.
    Reversible CmpOp pairs (LT↔GT, LE↔GE) are also normalized.

    SAFETY: per-block only. Never eliminates an instruction whose
    dest was already written in this block (loop writeback pattern).
    """
    eliminated = 0

    for bb in kernel.blocks:
        seen: dict[tuple, Value] = {}
        replacements: dict[int, Value] = {}
        written_ids: set[int] = set()
        new_insts = []

        def _key(op: Operand):
            if isinstance(op, Value):
                v = op
                while v.id in replacements:
                    v = replacements[v.id]
                return ('val', v.id)
            if isinstance(op, Const):
                return ('const', op.value)
            return ('other', id(op))

        for inst in bb.instructions:
            if isinstance(inst, BinInst):
                if isinstance(inst.lhs, Value) and inst.lhs.id in replacements:
                    inst.lhs = replacements[inst.lhs.id]
                if isinstance(inst.rhs, Value) and inst.rhs.id in replacements:
                    inst.rhs = replacements[inst.rhs.id]

                # Don't CSE if dest was already written (loop writeback)
                if inst.dest.id in written_ids:
                    new_insts.append(inst)
                    continue

                # Include dest TYPE in key — prevents merging int and float
                # variables that happen to have the same init value
                dest_type_key = str(inst.dest.ty)
                lk, rk = _key(inst.lhs), _key(inst.rhs)
                # Normalize commutative ops: canonical form is smaller key first
                if inst.op in _COMMUTATIVE_BINOPS and lk > rk:
                    lk, rk = rk, lk
                key = (inst.op, lk, rk, dest_type_key)
                if key in seen:
                    replacements[inst.dest.id] = seen[key]
                    eliminated += 1
                    continue

                seen[key] = inst.dest
                written_ids.add(inst.dest.id)

            elif isinstance(inst, CvtInst):
                # CSE for type conversions (e.g., cvt.u64.u32 of same source)
                if isinstance(inst.src, Value) and inst.src.id in replacements:
                    inst.src = replacements[inst.src.id]
                if inst.dest.id not in written_ids:
                    cvt_key = ('cvt', _key(inst.src), str(inst.dest.ty), str(inst.src.ty))
                    if cvt_key in seen:
                        replacements[inst.dest.id] = seen[cvt_key]
                        eliminated += 1
                        continue
                    seen[cvt_key] = inst.dest
                    written_ids.add(inst.dest.id)

            elif isinstance(inst, CmpInst):
                # Propagate replacements into operands
                if isinstance(inst.lhs, Value) and inst.lhs.id in replacements:
                    inst.lhs = replacements[inst.lhs.id]
                if isinstance(inst.rhs, Value) and inst.rhs.id in replacements:
                    inst.rhs = replacements[inst.rhs.id]

                # CSE: deduplicate identical comparisons within the block
                if inst.dest.id not in written_ids:
                    lk, rk = _key(inst.lhs), _key(inst.rhs)
                    op = inst.op
                    # Normalize commutative predicates (EQ/NE): smaller key first
                    if op in _CMP_COMMUTATIVE and lk > rk:
                        lk, rk = rk, lk
                    # Normalize swappable predicates (LT↔GT, LE↔GE): always lk<=rk form
                    elif op in _CMP_SWAP and lk > rk:
                        lk, rk = rk, lk
                        op = _CMP_SWAP[op]
                    cmp_key = ('cmp', op, lk, rk)
                    if cmp_key in seen:
                        replacements[inst.dest.id] = seen[cmp_key]
                        eliminated += 1
                        continue
                    seen[cmp_key] = inst.dest
                    written_ids.add(inst.dest.id)

            else:
                # Apply replacements to memory instructions
                if isinstance(inst, LoadInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                elif isinstance(inst, StoreInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                    if isinstance(inst.value, Value) and inst.value.id in replacements:
                        inst.value = replacements[inst.value.id]

            new_insts.append(inst)
        bb.instructions = new_insts

    return eliminated


def dead_block_elim(kernel: Kernel) -> int:
    """Remove basic blocks that are unreachable from the entry block.

    Unreachable blocks include 'after_break' and 'after_continue' stubs
    that the parser creates for dead code following a jump statement.
    Removing them reduces register pressure by eliminating instructions
    that otherwise contribute to the liveness analysis.

    SAFETY: Unreachable blocks cannot be the target of any live branch,
    so removing them cannot change observable behaviour.
    """
    if not kernel.blocks:
        return 0

    label_to_bb = {bb.label: bb for bb in kernel.blocks}
    reachable: set[str] = set()
    queue = [kernel.blocks[0].label]
    while queue:
        label = queue.pop()
        if label in reachable:
            continue
        reachable.add(label)
        bb = label_to_bb.get(label)
        if bb:
            t = bb.terminator
            if isinstance(t, BrTerm):
                queue.append(t.target)
            elif isinstance(t, CondBrTerm):
                queue.extend([t.true_bb, t.false_bb])

    removed = sum(1 for bb in kernel.blocks if bb.label not in reachable)
    kernel.blocks = [bb for bb in kernel.blocks if bb.label in reachable]
    return removed


def identity_fold(kernel: Kernel) -> int:
    """Copy propagation for add-zero patterns.

    Eliminates instructions of the form:
        add D, S, Const(0)   or   add D, Const(0), S
    when D has exactly one definition in the kernel (i.e., it is not a
    loop-writeback target). Replaces all uses of D with S globally.

    SAFETY RULE: Only single-definition Values are folded. The loop
    writeback mechanism creates a second definition of loop-carried
    Variables (def_count >= 2), so they are never touched here.
    This preserves the writeback semantics established in v0.4.

    NOTE: runs AFTER dead_block_elim so that dead-block instructions do
    not inflate def_count and prevent legitimate single-def folds.
    """
    # Count definitions of each Value ID across the entire kernel.
    def_count: dict[int, int] = {}
    for bb in kernel.blocks:
        for inst in bb.instructions:
            if hasattr(inst, 'dest') and inst.dest is not None:
                did = inst.dest.id
                def_count[did] = def_count.get(did, 0) + 1

    # Collect single-def add-zero copy instructions.
    copies: dict[int, Operand] = {}   # dest_id → source operand
    for bb in kernel.blocks:
        for inst in bb.instructions:
            if (isinstance(inst, BinInst)
                    and inst.op == BinOp.ADD
                    and def_count.get(inst.dest.id, 0) == 1):
                # add D, V, Const(0)  — V must be a Value, not another Const
                if isinstance(inst.lhs, Value) and isinstance(inst.rhs, Const) and inst.rhs.value == 0:
                    copies[inst.dest.id] = inst.lhs
                # add D, Const(0), V  — symmetric
                elif isinstance(inst.rhs, Value) and isinstance(inst.lhs, Const) and inst.lhs.value == 0:
                    copies[inst.dest.id] = inst.rhs

    if not copies:
        return 0

    def _resolve(op: Operand) -> Operand:
        if isinstance(op, Value) and op.id in copies:
            r = copies[op.id]
            # Follow chains (add A, B, 0; add C, A, 0 → C = B)
            seen: set[int] = {op.id}
            while isinstance(r, Value) and r.id in copies:
                if r.id in seen:
                    break
                seen.add(r.id)
                r = copies[r.id]
            return r
        return op

    eliminated = 0
    for bb in kernel.blocks:
        new_insts = []
        for inst in bb.instructions:
            if isinstance(inst, BinInst) and inst.dest.id in copies:
                eliminated += 1
                continue  # remove the copy instruction
            # Propagate replacements into all operand positions
            if isinstance(inst, BinInst):
                inst.lhs = _resolve(inst.lhs)
                inst.rhs = _resolve(inst.rhs)
            elif isinstance(inst, CmpInst):
                inst.lhs = _resolve(inst.lhs)
                inst.rhs = _resolve(inst.rhs)
            elif isinstance(inst, LoadInst):
                inst.addr = _resolve(inst.addr)
            elif isinstance(inst, StoreInst):
                inst.addr = _resolve(inst.addr)
                inst.value = _resolve(inst.value)
            elif isinstance(inst, CvtInst):
                inst.src = _resolve(inst.src)
            new_insts.append(inst)
        bb.instructions = new_insts

        # Fix terminator condition
        if isinstance(bb.terminator, CondBrTerm):
            resolved = _resolve(bb.terminator.cond)
            if resolved is not bb.terminator.cond:
                bb.terminator = CondBrTerm(
                    resolved, bb.terminator.true_bb, bb.terminator.false_bb)

    return eliminated


def dead_inst_elim(kernel: Kernel) -> int:
    """Remove instructions whose result is never consumed.

    Eliminates BinInst, CmpInst, and CvtInst where the dest Value is
    not referenced by any subsequent instruction or terminator. Iterates
    to fixpoint because eliminating one dead instruction may expose others.

    SAFETY: Only non-side-effecting instructions are removed. LoadInst,
    StoreInst, PrintfInst, CallInst, and ParamInst are never touched.
    CondBrTerm condition Values are counted as uses so branch predicates
    are never eliminated.
    """
    eliminated = 0
    changed = True
    while changed:
        changed = False

        # Collect every Value ID that is read as a source anywhere.
        used: set[int] = set()
        for bb in kernel.blocks:
            for inst in bb.instructions:
                for attr in ('lhs', 'rhs', 'src', 'addr', 'value'):
                    v = getattr(inst, attr, None)
                    if isinstance(v, Value):
                        used.add(v.id)
                # CallInst / PrintfInst args
                if hasattr(inst, 'args'):
                    for a in inst.args:
                        if isinstance(a, Value):
                            used.add(a.id)
            # Terminator condition
            t = bb.terminator
            if isinstance(t, CondBrTerm) and isinstance(t.cond, Value):
                used.add(t.cond.id)

        # Remove unreferenced non-side-effecting instructions.
        for bb in kernel.blocks:
            new_insts = []
            for inst in bb.instructions:
                if isinstance(inst, (BinInst, CmpInst, CvtInst)):
                    if inst.dest.id not in used:
                        eliminated += 1
                        changed = True
                        continue
                new_insts.append(inst)
            bb.instructions = new_insts

    return eliminated


def optimize(module: Module, verbose: bool = False) -> Module:
    """Run all optimization passes on the module.

    Pass order (designed to maximise cascading improvements):
      1. unroll_loops       — expose constants for folding
      2. constant_fold      — Const-op-Const, strength reduction
      3. cse                — eliminate duplicate expressions (commutative-
                              aware; includes CmpInst dedup since v0.7)
      4. dead_block_elim    — remove unreachable blocks before identity_fold
                              so dead-block defs don't inflate def_count
      5. identity_fold      — copy-propagate single-def add-zero patterns
      6. dead_inst_elim     — remove instructions whose results are unused
                              (iterates to fixpoint)
      7. cse (round 2)      — catch expressions newly exposed by passes 4–6
      8. identity_fold (2)  — propagate copies created by round-2 CSE
      9. dead_inst_elim (2) — remove instructions whose results are unused
                              after round-2 CSE+fold
    """
    from .unroll import unroll_loops

    for kernel in module.kernels:
        n_unroll = unroll_loops(kernel, max_unroll=16)
        n_fold = constant_fold(kernel)
        n_cse = cse(kernel)
        n_dbe = dead_block_elim(kernel)
        n_idf = identity_fold(kernel)
        n_die = dead_inst_elim(kernel)
        # Round 2: post-cleanup CSE catches expressions newly exposed above
        n_cse2 = cse(kernel)
        n_idf2 = identity_fold(kernel)
        n_die2 = dead_inst_elim(kernel)
        if verbose:
            total = n_unroll + n_fold + n_cse + n_dbe + n_idf + n_die + n_cse2 + n_idf2 + n_die2
            if total > 0:
                parts = []
                if n_unroll:       parts.append(f"{n_unroll} loops unrolled")
                if n_fold:         parts.append(f"{n_fold} constants folded")
                if n_cse:          parts.append(f"{n_cse} CSE eliminated")
                if n_dbe:          parts.append(f"{n_dbe} dead blocks removed")
                if n_idf:          parts.append(f"{n_idf} copies propagated")
                if n_die:          parts.append(f"{n_die} dead insts removed")
                if n_cse2:         parts.append(f"{n_cse2} CSE-2 eliminated")
                if n_idf2:         parts.append(f"{n_idf2} copies-2 propagated")
                if n_die2:         parts.append(f"{n_die2} dead-2 insts removed")
                print(f"[opt] {kernel.name}: {', '.join(parts)}")
    return module
