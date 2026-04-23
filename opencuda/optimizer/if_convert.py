"""If-conversion optimizer: convert load+mul+store diamonds to selp.

Matches the pattern

    if (cond) { out[idx] = load_val * const_true;  }
    else     { out[idx] = load_val * const_false; }

and rewrites to

    mult = selp(const_true, const_false, cond);
    out[idx] = load_val * mult;

This hits ~21 SASS instructions vs ~4 for the branchless form. Today this
fires on exactly one kernel in the test corpus (`cond_test` in
test_gpu_e2e.py). A two-pass measurement (2026-04-23) across all 81 test
kernels showed the earlier "single-arm inline" and "MUL(0,X) cleanup"
passes firing zero times — they were written too narrowly to match
anything OpenCUDA actually emits after the main optimize() pipeline. Both
were removed; broader if-conversion should live in a rewrite rather than
be retrofitted here.
"""
from __future__ import annotations
from opencuda.ir.nodes import (
    Kernel, BasicBlock, Value, Const,
    BinInst, BinOp, CmpInst, CvtInst, LoadInst, StoreInst, SelectInst,
    CondBrTerm, BrTerm, RetTerm,
)
from opencuda.ir.types import ScalarTy, ScalarType, FLOAT


def _match_arm(bb: BasicBlock):
    """Check if a basic block is a diamond arm with load+mul+store pattern.

    Returns (store_addr_chain, load_addr_chain, mul_const, store_inst) or None.
    The addr_chains are tuples of (base_value_id, offset_value_id) for matching.
    """
    if not isinstance(bb.terminator, BrTerm):
        return None

    instrs = bb.instructions
    # Find the store (must be last real instruction)
    store = None
    mul_inst = None
    load = None
    for inst in reversed(instrs):
        if isinstance(inst, StoreInst) and store is None:
            store = inst
        elif isinstance(inst, BinInst) and inst.op == BinOp.MUL and mul_inst is None:
            mul_inst = inst
        elif isinstance(inst, LoadInst) and load is None:
            load = inst

    if not (store and mul_inst and load):
        return None

    # The mul should produce the stored value
    if not (isinstance(store.value, Value) and store.value.id == mul_inst.dest.id):
        return None

    # One operand of mul should be the loaded value, the other a constant
    mul_const = None
    if isinstance(mul_inst.lhs, Value) and mul_inst.lhs.id == load.dest.id and isinstance(mul_inst.rhs, Const):
        mul_const = mul_inst.rhs
    elif isinstance(mul_inst.rhs, Value) and mul_inst.rhs.id == load.dest.id and isinstance(mul_inst.lhs, Const):
        mul_const = mul_inst.lhs

    if mul_const is None:
        return None

    # Extract the base+offset pattern for addresses
    # Look for BinInst(ADD, base, offset) patterns for both store and load addresses
    def _addr_pattern(addr_val, instrs):
        """Find the (base_param_id, index_val_id) for an address computation."""
        if not isinstance(addr_val, Value):
            return None
        for inst in instrs:
            if isinstance(inst, BinInst) and inst.dest.id == addr_val.id and inst.op == BinOp.ADD:
                base = inst.lhs
                offset = inst.rhs
                if isinstance(base, Value) and isinstance(offset, Value):
                    return (base.id, offset.id)
        return None

    store_pattern = _addr_pattern(store.addr, instrs)
    load_pattern = _addr_pattern(load.addr, instrs)

    return (store_pattern, load_pattern, mul_const, store, load, mul_inst)


def if_convert_diamonds(kernel: Kernel) -> Kernel:
    """Convert if/else diamonds with load+mul+store to selp-based code."""
    bb_map = {bb.label: bb for bb in kernel.blocks}
    blocks_to_remove = set()
    modified = False

    for bb in kernel.blocks:
        if not isinstance(bb.terminator, CondBrTerm):
            continue

        term = bb.terminator
        t_bb = bb_map.get(term.true_bb)
        f_bb = bb_map.get(term.false_bb)
        if not (t_bb and f_bb):
            continue

        # Both arms must branch to the same merge block
        if not (isinstance(t_bb.terminator, BrTerm) and isinstance(f_bb.terminator, BrTerm)):
            continue
        if t_bb.terminator.target != f_bb.terminator.target:
            continue
        merge_label = t_bb.terminator.target

        # Match both arms
        t_match = _match_arm(t_bb)
        f_match = _match_arm(f_bb)
        if not (t_match and f_match):
            continue

        t_store_pat, t_load_pat, t_const, t_store, t_load, t_mul = t_match
        f_store_pat, f_load_pat, f_const, f_store, f_load, f_mul = f_match

        # Both arms must have the same type of mul constant
        if type(t_const.value) != type(f_const.value):
            continue

        # Check if the parent block already has a load from the same source
        # (the comparison load). We need to find a LoadInst in bb whose
        # result feeds into the CmpInst that drives the CondBrTerm.
        parent_load = None
        parent_load_addr = None
        cmp_inst = None
        for inst in bb.instructions:
            if isinstance(inst, CmpInst):
                cmp_inst = inst
            if isinstance(inst, LoadInst):
                parent_load = inst
                # Find the addr pattern
                for inst2 in bb.instructions:
                    if (isinstance(inst2, BinInst) and inst2.dest.id == inst.addr.id
                            and inst2.op == BinOp.ADD):
                        parent_load_addr = inst2
                        break

        if not (parent_load and cmp_inst):
            continue

        # Verify the cmp uses the parent load result
        cmp_uses_load = (
            (isinstance(cmp_inst.lhs, Value) and cmp_inst.lhs.id == parent_load.dest.id) or
            (isinstance(cmp_inst.rhs, Value) and cmp_inst.rhs.id == parent_load.dest.id)
        )
        if not cmp_uses_load:
            continue

        # Build the replacement: selp + mul + store
        # The condition is the CmpInst result
        cond_val = cmp_inst.dest

        # Determine result type from the mul constant
        if isinstance(t_const.value, float):
            result_ty = ScalarTy(ScalarType.FLOAT)
        else:
            result_ty = ScalarTy(ScalarType.INT32)

        # Create SelectInst: mult = selp(true_const, false_const, cond)
        select_dest = Value(name='_selp_mult', ty=result_ty, id=max(
            inst.dest.id for inst in bb.instructions if hasattr(inst, 'dest')
        ) + 100)
        select_inst = SelectInst(
            dest=select_dest,
            cond=cond_val,
            true_val=t_const,
            false_val=f_const,
        )

        # Create MulInst: result = load_val * mult
        mul_dest = Value(name='_selp_result', ty=result_ty, id=select_dest.id + 1)
        mul_inst = BinInst(
            dest=mul_dest,
            op=BinOp.MUL,
            lhs=parent_load.dest,
            rhs=select_dest,
        )

        # We need the output address. Recompute it from the parent block's
        # available values (out + idx*4). The address computation is the same
        # in both arms, so we can copy it from either arm.
        # Copy the address computation instructions from the true arm
        # (everything except the load, mul, and store)
        addr_instrs = []
        for inst in t_bb.instructions:
            if inst is t_load or inst is t_mul or inst is t_store:
                continue
            # Check if this computes the store address
            addr_instrs.append(inst)

        # Create StoreInst using the true arm's store address
        store_inst = StoreInst(
            addr=t_store.addr,
            value=mul_dest,
        )

        # Replace: add selp + mul + addr_calc + store to parent block,
        # remove the CondBrTerm, add BrTerm to merge (or outer merge)
        bb.instructions.append(select_inst)
        bb.instructions.append(mul_inst)
        bb.instructions.extend(addr_instrs)
        bb.instructions.append(store_inst)

        # Change terminator to branch directly to the merge block's target
        # (skip the inner merge block if it just branches to outer merge)
        merge_bb = bb_map.get(merge_label)
        if (merge_bb and len(merge_bb.instructions) == 0
                and isinstance(merge_bb.terminator, BrTerm)):
            # Inner merge is empty, branch to its target
            bb.terminator = BrTerm(target=merge_bb.terminator.target)
            blocks_to_remove.add(merge_label)
        else:
            bb.terminator = BrTerm(target=merge_label)

        blocks_to_remove.add(term.true_bb)
        blocks_to_remove.add(term.false_bb)
        modified = True

    if modified:
        kernel.blocks = [bb for bb in kernel.blocks if bb.label not in blocks_to_remove]

    return kernel
