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
Pass 7: LICM — Loop-Invariant Code Motion (v0.8). Conservative:
        only hoists pure BinInst/CvtInst whose operands are all defined
        outside the loop, with def_count==1 (not a writeback target).
        Never hoists memory ops, calls, side-effecting instructions, or
        loop-condition comparisons.

SAFETY RULE for passes 1–2: Replacements never cross basic block boundaries.
This prevents the loop writeback bug where a variable initialized
in the entry block (float sum = 0) gets replaced by Const(0) in
the loop body, causing the loop condition to never see updates.

SAFETY RULE for pass 4 (identity_fold): only propagates single-definition
Values (def_count == 1). Loop-writeback Values are redefined each iteration
and therefore have def_count >= 2 — they are never touched.

SAFETY RULE for pass 7 (licm): only hoists instructions with def_count==1.
Values that appear in CondBrTerm conditions (predicates) are never hoisted
even if technically invariant, since they guard the loop exit.
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


def _mask_int(value: int, ty) -> int:
    """Mask an integer fold result to the type's bit width with correct sign.

    Python integers are unbounded, so constant folding can produce values
    that are out of range for the PTX type (e.g. 1 << 31 = 2147483648 for
    INT32, but INT32_MAX is 2147483647). Without masking the emitter writes
    a literal that ptxas rejects for signed types.
    """
    from ..ir.types import ScalarTy
    if not isinstance(ty, ScalarTy) or ty.is_float:
        return value
    bits = ty.size * 8
    mask = (1 << bits) - 1
    v = int(value) & mask
    # Sign-extend for signed types: if the high bit is set, subtract 2^bits
    if ty.is_signed and (v >> (bits - 1)):
        v -= (1 << bits)
    return v


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
        if op == BinOp.DIV and b != 0:
            if is_float:
                return a / b
            # C truncated division (toward zero), not Python floor division.
            # Example: -7 / 2 = -3 in C, but -7 // 2 = -4 in Python.
            q = abs(a) // abs(b)
            return q if (a >= 0) == (b >= 0) else -q
        if op == BinOp.MOD and b != 0:
            if is_float:
                return a % b
            # C truncated remainder: result has sign of dividend.
            # Example: -7 % 2 = -1 in C, but -7 % 2 = 1 in Python.
            q = abs(a) // abs(b)
            rem = abs(a) - q * abs(b)
            return rem if a >= 0 else -rem
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
                        # Mask to type width: Python ints are unbounded but PTX
                        # types are not (e.g. 1 << 31 = 2147483648, out of s32 range).
                        masked = _mask_int(result, inst.dest.ty)
                        inst.lhs = Const(inst.dest.ty, masked)
                        inst.rhs = Const(inst.dest.ty, 0)
                        inst.op = BinOp.ADD
                    folded += 1
                    # Keep the instruction (it's now simpler but still writes dest)

                # Identity: x * 1 → x.  Emit as add-zero so identity_fold eliminates it.
                elif not is_float and inst.op == BinOp.MUL and rv == 1 and isinstance(inst.lhs, Value):
                    inst.op = BinOp.ADD
                    inst.rhs = Const(inst.dest.ty, 0)
                    folded += 1
                elif not is_float and inst.op == BinOp.MUL and lv == 1 and isinstance(inst.rhs, Value):
                    inst.op = BinOp.ADD
                    inst.lhs = inst.rhs
                    inst.rhs = Const(inst.dest.ty, 0)
                    folded += 1

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
            elif isinstance(inst, CvtInst):
                src = _resolve(inst.src)
                inst.src = src
                # Fold CvtInst whose source is a compile-time constant.
                # Convert the constant value to the destination type and
                # replace with a simpler add-zero instruction.
                if isinstance(src, Const):
                    dest_ty = inst.dest.ty
                    dest_is_float = isinstance(dest_ty, ScalarTy) and dest_ty.is_float
                    src_is_float = isinstance(src.ty, ScalarTy) and src.ty.is_float
                    try:
                        if dest_is_float:
                            folded_val = float(src.value)
                        else:
                            raw = float(src.value) if src_is_float else int(src.value)
                            folded_val = _mask_int(int(raw), dest_ty)
                        inst.lhs = Const(dest_ty, folded_val)
                        inst.rhs = Const(dest_ty, 0.0 if dest_is_float else 0)
                        # Replace with BinInst ADD so the rest of the pipeline
                        # can fold it further via the replacement map.
                        new_insts.append(
                            BinInst(inst.dest, BinOp.ADD, inst.lhs, inst.rhs))
                        folded += 1
                        continue
                    except Exception:
                        pass
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

    SAFETY: Never eliminates an instruction whose dest is writeback-carried
    (global def_count >= 2).  Two identically-valued initializers like
    `int sum = 0; int i = 0;` emit identical `add v, 0, 0` instructions in
    the same entry block.  Without this guard, CSE merges their dest Values,
    aliasing sum ↔ i.  The CSE global-replacement sweep then rewrites all
    uses of i (the loop counter) with sum's register, silently breaking the
    loop condition while leaving the for_inc writeback with def_count==1 so
    that identity_fold subsequently deletes it.

    Cross-block correctness: after all local passes, a second sweep applies
    accumulated replacements to operands in all blocks.  This handles the
    case where block A eliminates def X (keeping canonical Y) but block B
    (which post-dominates A) still references X.
    """
    # Pre-compute global def_count so we can protect writeback-carried Values.
    _global_def_count: dict[int, int] = {}
    for _bb in kernel.blocks:
        for _inst in _bb.instructions:
            if hasattr(_inst, 'dest') and _inst.dest is not None:
                _did = _inst.dest.id
                _global_def_count[_did] = _global_def_count.get(_did, 0) + 1

    eliminated = 0
    global_replacements: dict[int, Value] = {}

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

                # Don't CSE if dest was already written (loop writeback in THIS block)
                if inst.dest.id in written_ids:
                    new_insts.append(inst)
                    continue

                # Don't CSE if dest is writeback-carried (defined in multiple blocks).
                # Two identically-initialised writeback vars (e.g. sum=0, i=0) emit
                # identical `add v, 0, 0` in the same entry block.  Merging them via
                # global_replacements would alias their registers and silently corrupt
                # the loop counter / accumulator semantics downstream.
                if _global_def_count.get(inst.dest.id, 0) >= 2:
                    written_ids.add(inst.dest.id)
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
                if inst.dest.id not in written_ids and _global_def_count.get(inst.dest.id, 0) < 2:
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
                if inst.dest.id not in written_ids and _global_def_count.get(inst.dest.id, 0) < 2:
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

            elif isinstance(inst, CallInst):
                # CSE for pure zero-arg CUDA builtins (threadIdx/blockIdx/blockDim).
                # These return a constant value per-thread for the kernel's lifetime,
                # so duplicate reads within the same block are always redundant.
                # We do NOT CSE arbitrary CallInsts (they may have side effects).
                _PURE_BUILTINS = frozenset({
                    'threadIdx.x', 'threadIdx.y', 'threadIdx.z',
                    'blockIdx.x',  'blockIdx.y',  'blockIdx.z',
                    'blockDim.x',  'blockDim.y',  'blockDim.z',
                    'gridDim.x',   'gridDim.y',   'gridDim.z',
                })
                if (inst.func in _PURE_BUILTINS
                        and inst.dest is not None
                        and inst.dest.id not in written_ids
                        and _global_def_count.get(inst.dest.id, 0) < 2):
                    call_key = ('pure_call', inst.func)
                    if call_key in seen:
                        replacements[inst.dest.id] = seen[call_key]
                        eliminated += 1
                        continue
                    seen[call_key] = inst.dest
                    written_ids.add(inst.dest.id)

            else:
                # Apply replacements to memory and side-effecting instructions
                if isinstance(inst, LoadInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                elif isinstance(inst, StoreInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                    if isinstance(inst.value, Value) and inst.value.id in replacements:
                        inst.value = replacements[inst.value.id]
                elif isinstance(inst, (PrintfInst, CallInst)):
                    inst.args = [replacements[a] if isinstance(a, Value) and a.id in replacements else a
                                 for a in inst.args]

            new_insts.append(inst)
        bb.instructions = new_insts
        global_replacements.update(replacements)

    # Second pass: propagate replacements to any cross-block uses that the
    # per-block loop could not reach (e.g. block A eliminates def X → Y, but
    # block B still references X).
    if global_replacements:
        def _chase(v: Value) -> Value:
            while v.id in global_replacements:
                v = global_replacements[v.id]
            return v

        for bb in kernel.blocks:
            for inst in bb.instructions:
                if isinstance(inst, BinInst):
                    if isinstance(inst.lhs, Value):
                        inst.lhs = _chase(inst.lhs)
                    if isinstance(inst.rhs, Value):
                        inst.rhs = _chase(inst.rhs)
                elif isinstance(inst, CmpInst):
                    if isinstance(inst.lhs, Value):
                        inst.lhs = _chase(inst.lhs)
                    if isinstance(inst.rhs, Value):
                        inst.rhs = _chase(inst.rhs)
                elif isinstance(inst, CvtInst):
                    if isinstance(inst.src, Value):
                        inst.src = _chase(inst.src)
                elif isinstance(inst, LoadInst):
                    if isinstance(inst.addr, Value):
                        inst.addr = _chase(inst.addr)
                elif isinstance(inst, StoreInst):
                    if isinstance(inst.addr, Value):
                        inst.addr = _chase(inst.addr)
                    if isinstance(inst.value, Value):
                        inst.value = _chase(inst.value)
                elif isinstance(inst, CallInst):
                    inst.args = [_chase(a) if isinstance(a, Value) else a
                                 for a in inst.args]
                elif isinstance(inst, PrintfInst):
                    inst.args = [_chase(a) if isinstance(a, Value) else a
                                 for a in inst.args]
            t = bb.terminator
            if isinstance(t, CondBrTerm) and isinstance(t.cond, Value):
                new_cond = _chase(t.cond)
                if new_cond is not t.cond:
                    bb.terminator = CondBrTerm(new_cond, t.true_bb, t.false_bb)

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
    copies: dict[int, Operand] = {}   # dest_id → source operand (Value or Const)
    # Ops where `D = X op Const(0)` is equivalent to `D = X`
    _ZERO_RHS_IDENTITY = frozenset({
        BinOp.ADD, BinOp.SUB, BinOp.SHL, BinOp.SHR,
        BinOp.XOR, BinOp.OR,
    })
    # Ops where `D = Const(0) op X` is also equivalent to `D = X` (commutative zero)
    _ZERO_LHS_IDENTITY = frozenset({BinOp.ADD, BinOp.XOR, BinOp.OR})

    for bb in kernel.blocks:
        for inst in bb.instructions:
            if (isinstance(inst, BinInst)
                    and def_count.get(inst.dest.id, 0) == 1):
                # D = X op Const(0) — X may be Value or Const (e.g. result = -1)
                if (inst.op in _ZERO_RHS_IDENTITY
                        and isinstance(inst.rhs, Const) and inst.rhs.value == 0):
                    copies[inst.dest.id] = inst.lhs
                # D = Const(0) op X — symmetric for commutative ops
                elif (inst.op in _ZERO_LHS_IDENTITY
                        and isinstance(inst.lhs, Const) and inst.lhs.value == 0):
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
            elif isinstance(inst, CallInst):
                inst.args = [_resolve(a) for a in inst.args]
            elif isinstance(inst, PrintfInst):
                inst.args = [_resolve(a) for a in inst.args]
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


# ---------------------------------------------------------------------------
# LICM helpers
# ---------------------------------------------------------------------------

def _bb_successors(bb: BasicBlock) -> list[str]:
    t = bb.terminator
    if isinstance(t, BrTerm):
        return [t.target]
    if isinstance(t, CondBrTerm):
        return [t.true_bb, t.false_bb]
    return []


class _Loop:
    """A natural loop detected by back-edge analysis."""
    __slots__ = ('header', 'preheader', 'body', 'backedge_src')

    def __init__(self, header: str, preheader: str,
                 body: frozenset, backedge_src: str) -> None:
        self.header = header
        self.preheader = preheader
        self.body = body          # frozenset of block labels
        self.backedge_src = backedge_src


def _find_loops(kernel: Kernel) -> list[_Loop]:
    """Detect natural loops by finding back edges via DFS.

    A back edge is an edge (u → v) where v is an *ancestor* of u in the DFS
    spanning tree (i.e. v is on the current DFS stack when we visit u).  This
    is the standard definition and correctly rejects non-loop forward/cross
    edges even when the block list ordering would misclassify them.

    Example: the parser emits outer-merge blocks before inner content, so
    block-list-position heuristics create false back edges in nested if-else
    chains.  DFS-based detection avoids this.

    For each back edge (u → header):
      • loop body: reverse BFS from u back to header (exclusive)
      • preheader: the unique non-loop predecessor of header (required for
        safe hoisting); skipped if ambiguous

    Returns one _Loop per back edge.
    """
    if not kernel.blocks:
        return []

    label_to_bb = {bb.label: bb for bb in kernel.blocks}
    entry = kernel.blocks[0].label

    # Build predecessor map
    pred_map: dict[str, list[str]] = {bb.label: [] for bb in kernel.blocks}
    for bb in kernel.blocks:
        for s in _bb_successors(bb):
            if s in pred_map:
                pred_map[s].append(bb.label)

    # DFS to find back edges (u → v where v is on the current stack)
    back_edges: list[tuple[str, str]] = []  # (src, header)
    visited: set[str] = set()
    on_stack: set[str] = set()

    def dfs(lbl: str) -> None:
        if lbl in visited:
            return
        visited.add(lbl)
        on_stack.add(lbl)
        bb = label_to_bb.get(lbl)
        if bb:
            for succ in _bb_successors(bb):
                if succ not in label_to_bb:
                    continue
                if succ in on_stack:
                    back_edges.append((lbl, succ))  # back edge
                elif succ not in visited:
                    dfs(succ)
        on_stack.discard(lbl)

    dfs(entry)

    loops: list[_Loop] = []
    seen: set[tuple[str, str]] = set()

    for backedge_src, header in back_edges:
        edge = (backedge_src, header)
        if edge in seen:
            continue
        seen.add(edge)

        # Loop body: reverse BFS from backedge_src back to header
        body: set[str] = {header}
        work = [backedge_src]
        while work:
            cur = work.pop()
            if cur in body:
                continue
            body.add(cur)
            for p in pred_map.get(cur, []):
                if p not in body:
                    work.append(p)

        # Preheader: the unique non-loop predecessor of the header
        non_loop_preds = [
            p for p in pred_map.get(header, [])
            if p not in body
        ]
        if len(non_loop_preds) != 1:
            continue  # no unique preheader — skip
        preheader = non_loop_preds[0]

        loops.append(_Loop(
            header=header,
            preheader=preheader,
            body=frozenset(body),
            backedge_src=backedge_src,
        ))

    return loops


def licm(kernel: Kernel) -> int:
    """Loop-Invariant Code Motion (conservative).

    For each natural loop, hoists pure instructions (BinInst, CvtInst) to the
    loop preheader when ALL of the following hold:

      1. All source operands are loop-invariant (defined outside the loop or
         by a previously hoisted instruction).
      2. The destination Value has exactly one definition in the kernel
         (def_count == 1) — excludes writeback-carried loop variables.
      3. The instruction is not a CmpInst whose result feeds a CondBrTerm
         within the loop (would hoist the loop's exit condition).
      4. Never hoists LoadInst, StoreInst, CallInst, PrintfInst, ParamInst.

    Uses an inner fixpoint loop to handle chains of invariant instructions
    (where hoisting A makes B's operands invariant, enabling B to be hoisted).

    Returns total number of instructions hoisted across all loops.
    """
    loops = _find_loops(kernel)
    if not loops:
        return 0

    label_to_bb = {bb.label: bb for bb in kernel.blocks}
    label_to_idx = {bb.label: i for i, bb in enumerate(kernel.blocks)}

    # def_count: number of times each Value ID is written across the whole kernel
    def_count: dict[int, int] = {}
    for bb in kernel.blocks:
        for inst in bb.instructions:
            if hasattr(inst, 'dest') and inst.dest is not None:
                did = inst.dest.id
                def_count[did] = def_count.get(did, 0) + 1

    # cond_ids: Value IDs used directly as CondBrTerm conditions in any block.
    # Never hoist these — they guard loop exits.
    cond_ids: set[int] = set()
    for bb in kernel.blocks:
        t = bb.terminator
        if isinstance(t, CondBrTerm) and isinstance(t.cond, Value):
            cond_ids.add(t.cond.id)

    total_hoisted = 0

    for loop in loops:
        preheader_bb = label_to_bb.get(loop.preheader)
        if preheader_bb is None:
            continue

        # Values with ANY definition inside the loop are NOT invariant.
        # This is critical: loop-carried variables (i, sum, etc.) are defined
        # both in the preheader (initial value) AND inside the loop body
        # (writeback in for_inc / while_body). If we only checked "defined
        # outside", they'd incorrectly appear invariant.
        loop_defs: set[int] = set()
        for bb in kernel.blocks:
            if bb.label in loop.body:
                for inst in bb.instructions:
                    if hasattr(inst, 'dest') and inst.dest is not None:
                        loop_defs.add(inst.dest.id)

        # Seed: all Values defined outside this loop AND never defined inside
        inv_ids: set[int] = set()
        for bb in kernel.blocks:
            if bb.label not in loop.body:
                for inst in bb.instructions:
                    if hasattr(inst, 'dest') and inst.dest is not None:
                        vid = inst.dest.id
                        if vid not in loop_defs:
                            inv_ids.add(vid)

        def _is_inv(op: Operand) -> bool:
            return isinstance(op, Const) or (isinstance(op, Value) and op.id in inv_ids)

        # Fixpoint: repeat until no more instructions can be hoisted
        changed = True
        while changed:
            changed = False

            # Process loop body blocks in block-list order (approximates topo order)
            for lbl in sorted(loop.body, key=lambda l: label_to_idx.get(l, 0)):
                if lbl == loop.header:
                    continue  # do not hoist from the loop header itself
                bb = label_to_bb.get(lbl)
                if bb is None:
                    continue

                new_insts = []
                for inst in bb.instructions:
                    # Only consider pure BinInst and CvtInst
                    if not isinstance(inst, (BinInst, CvtInst)):
                        new_insts.append(inst)
                        continue

                    # Dest must have exactly one definition (not writeback-carried)
                    if def_count.get(inst.dest.id, 0) != 1:
                        new_insts.append(inst)
                        continue

                    # Never hoist a value used as a branch condition
                    if inst.dest.id in cond_ids:
                        new_insts.append(inst)
                        continue

                    # All operands must be loop-invariant
                    if isinstance(inst, BinInst):
                        operands_inv = _is_inv(inst.lhs) and _is_inv(inst.rhs)
                    else:  # CvtInst
                        operands_inv = _is_inv(inst.src)

                    if not operands_inv:
                        new_insts.append(inst)
                        continue

                    # Hoist: append to preheader's instruction list (before terminator)
                    preheader_bb.instructions.append(inst)
                    inv_ids.add(inst.dest.id)
                    total_hoisted += 1
                    changed = True

                bb.instructions = new_insts

    return total_hoisted


def thread_empty_blocks(kernel: Kernel) -> int:
    """Branch threading: forward unconditional branches through empty blocks.

    A block is *transparent* if it has no instructions and a single
    unconditional BrTerm.  Any predecessor that targets a transparent block
    can skip directly to its ultimate destination, shortening branch chains
    and allowing the transparent blocks to be removed entirely.

    Algorithm
    ---------
    1. Build a forwarding map: transparent_label → forwarded_target.
       Follow chains (A→B→C where B and C are also transparent) with
       cycle detection to avoid infinite loops.
    2. Rewrite every BrTerm and CondBrTerm in every block to use the
       forwarded target instead of the original.
       Special case: if CondBrTerm.true_bb == CondBrTerm.false_bb after
       forwarding, replace with a plain BrTerm (dead-branch folding).
    3. Remove transparent blocks that are no longer needed.
       Exception: the entry block (first block) is never removed even if
       it is transparent, because the kernel must start execution there.

    Returns the number of transparent blocks threaded (removed).

    SAFETY: Never threads across a block that has instructions (it may have
    side effects).  The entry block is never removed.  Loop back-edges are
    preserved because the loop header always has at least a CmpInst.
    """
    if not kernel.blocks:
        return 0

    entry_label = kernel.blocks[0].label

    # Step 1: build forwarding map (only for non-entry transparent blocks)
    label_to_bb = {bb.label: bb for bb in kernel.blocks}

    def _forward(lbl: str) -> str:
        """Follow transparent blocks to their ultimate target."""
        visited: set[str] = set()
        while True:
            if lbl in visited:
                return lbl  # cycle — bail out
            bb = label_to_bb.get(lbl)
            if bb is None:
                return lbl
            if lbl == entry_label:
                return lbl
            if len(bb.instructions) == 0 and isinstance(bb.terminator, BrTerm):
                visited.add(lbl)
                lbl = bb.terminator.target
            else:
                return lbl

    # Collect transparent block labels (before rewriting, so the map is stable)
    transparent: set[str] = set()
    for bb in kernel.blocks:
        if bb.label != entry_label and len(bb.instructions) == 0 and isinstance(bb.terminator, BrTerm):
            transparent.add(bb.label)

    if not transparent:
        return 0

    # Precompute forwarding targets for all transparent blocks
    fwd: dict[str, str] = {lbl: _forward(lbl) for lbl in transparent}

    def _fwd(lbl: str) -> str:
        return fwd.get(lbl, lbl)

    # Step 2: rewrite branch targets in all blocks
    for bb in kernel.blocks:
        t = bb.terminator
        if isinstance(t, BrTerm):
            new_tgt = _fwd(t.target)
            if new_tgt != t.target:
                bb.terminator = BrTerm(new_tgt)
        elif isinstance(t, CondBrTerm):
            new_true = _fwd(t.true_bb)
            new_false = _fwd(t.false_bb)
            if new_true != t.true_bb or new_false != t.false_bb:
                if new_true == new_false:
                    # Both arms go to same target — fold to unconditional branch
                    bb.terminator = BrTerm(new_true)
                else:
                    bb.terminator = CondBrTerm(t.cond, new_true, new_false)

    # Step 3: remove transparent blocks that now have no in-edges.
    # Rebuild pred map after rewriting.
    referenced: set[str] = {entry_label}
    for bb in kernel.blocks:
        t = bb.terminator
        if isinstance(t, BrTerm):
            referenced.add(t.target)
        elif isinstance(t, CondBrTerm):
            referenced.add(t.true_bb)
            referenced.add(t.false_bb)

    removed = 0
    new_blocks = []
    for bb in kernel.blocks:
        if bb.label in transparent and bb.label not in referenced:
            removed += 1
        else:
            new_blocks.append(bb)
    kernel.blocks = new_blocks

    return removed


def optimize(module: Module, verbose: bool = False,
             debug_verify: bool = False) -> Module:
    """Run all optimization passes on the module.

    Pass order (designed to maximise cascading improvements):
      1. unroll_loops         — expose constants for folding
      2. constant_fold        — Const-op-Const, strength reduction
      3. cse                  — eliminate duplicate expressions (commutative-
                                aware; includes CmpInst dedup since v0.7;
                                writeback-carried guard since v0.11)
      4. dead_block_elim      — remove unreachable blocks before identity_fold
                                so dead-block defs don't inflate def_count
      5. identity_fold        — copy-propagate single-def add-zero patterns
      5b. constant_fold (2)   — fold newly-exposed Const-op-Const after
                                identity_fold (e.g. 0-7 folds neg7, identity_fold
                                propagates Const(-7) into div, fold-2 finishes it)
      6. dead_inst_elim       — remove instructions whose results are unused
                                (iterates to fixpoint)
      7. licm                 — hoist loop-invariant pure BinInst/CvtInst to
                                loop preheaders (conservative; v0.8)
      8. cse (round 2)        — catch expressions newly exposed by passes 4–7
      9. identity_fold (2)    — propagate copies created by round-2 CSE
      9b. constant_fold (3)   — fold constants newly exposed by identity_fold-2
     10. dead_inst_elim (2)   — remove instructions whose results are unused
                                after LICM + round-2 CSE+fold
     11. thread_empty_blocks  — forward unconditional branches through empty
                                blocks, eliminating ~22% of all blocks (v0.11)

    Parameters
    ----------
    debug_verify : bool
        If True, run verify_kernel after every pass and raise AssertionError
        on any violation.  Off by default — use in tests and debugging, not
        in production builds.
    """
    from .unroll import unroll_loops

    if debug_verify:
        from ..ir.verify_ir import verify_kernel as _verify

        def _gate(kernel, pass_name: str) -> None:
            errs = _verify(kernel, check_reachability=False)
            if errs:
                raise AssertionError(
                    f'IR violation after {pass_name} in {kernel.name!r}:\n'
                    + '\n'.join(errs))
    else:
        def _gate(kernel, pass_name: str) -> None:  # type: ignore[misc]
            pass

    for kernel in module.kernels:
        n_unroll = unroll_loops(kernel, max_unroll=16)
        _gate(kernel, 'unroll_loops')
        n_fold = constant_fold(kernel)
        _gate(kernel, 'constant_fold')
        n_cse = cse(kernel)
        _gate(kernel, 'cse')
        n_dbe = dead_block_elim(kernel)
        _gate(kernel, 'dead_block_elim')
        n_idf = identity_fold(kernel)
        _gate(kernel, 'identity_fold')
        # constant_fold-2: finish folding expressions where identity_fold just
        # propagated a Const into a BinInst (e.g. neg7_val → Const(-7) in div)
        n_fold2 = constant_fold(kernel)
        _gate(kernel, 'constant_fold-2')
        n_die = dead_inst_elim(kernel)
        _gate(kernel, 'dead_inst_elim')
        n_licm = licm(kernel)
        _gate(kernel, 'licm')
        # Round 2: post-LICM CSE catches newly exposed duplicates
        n_cse2 = cse(kernel)
        _gate(kernel, 'cse-2')
        n_idf2 = identity_fold(kernel)
        _gate(kernel, 'identity_fold-2')
        # constant_fold-3: catch constants exposed by identity_fold-2
        n_fold3 = constant_fold(kernel)
        _gate(kernel, 'constant_fold-3')
        n_die2 = dead_inst_elim(kernel)
        _gate(kernel, 'dead_inst_elim-2')
        n_teb = thread_empty_blocks(kernel)
        _gate(kernel, 'thread_empty_blocks')
        if verbose:
            total = (n_unroll + n_fold + n_fold2 + n_fold3 + n_cse + n_dbe
                     + n_idf + n_die + n_licm + n_cse2 + n_idf2 + n_die2 + n_teb)
            if total > 0:
                parts = []
                if n_unroll:                 parts.append(f"{n_unroll} loops unrolled")
                if n_fold + n_fold2 + n_fold3:
                    parts.append(f"{n_fold + n_fold2 + n_fold3} constants folded")
                if n_cse:                    parts.append(f"{n_cse} CSE eliminated")
                if n_dbe:                    parts.append(f"{n_dbe} dead blocks removed")
                if n_idf:                    parts.append(f"{n_idf} copies propagated")
                if n_die:                    parts.append(f"{n_die} dead insts removed")
                if n_licm:                   parts.append(f"{n_licm} LICM hoisted")
                if n_cse2:                   parts.append(f"{n_cse2} CSE-2 eliminated")
                if n_idf2:                   parts.append(f"{n_idf2} copies-2 propagated")
                if n_die2:                   parts.append(f"{n_die2} dead-2 insts removed")
                if n_teb:                    parts.append(f"{n_teb} empty blocks threaded")
                print(f"[opt] {kernel.name}: {', '.join(parts)}")
    return module
