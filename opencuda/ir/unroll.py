"""
Loop unrolling pass for OpenCUDA IR.

Detects for-loops with compile-time-known trip counts and unrolls them.
Handles loop-carried variables (accumulators) by chaining each iteration's
output to the next iteration's input.

The key insight: in `for (int k=0; k<16; k++) { sum += f(k); }`, the
variable `sum` is "loop-carried" — each iteration reads the previous
iteration's output. The unroller must connect these across copies.
"""

from __future__ import annotations
from ..ir.nodes import (Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst,
                         CallInst, ParamInst, CvtInst, PrintfInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)


def _find_unrollable_loops(kernel: Kernel) -> list[dict]:
    """Find for-loops with constant bounds that can be unrolled."""
    loops = []
    blocks_by_label = {bb.label: bb for bb in kernel.blocks}

    for bb in kernel.blocks:
        if not isinstance(getattr(bb, 'terminator', None), CondBrTerm):
            continue

        cond_bb = bb
        cmp_inst = None
        for inst in cond_bb.instructions:
            if isinstance(inst, CmpInst):
                cmp_inst = inst

        if cmp_inst is None:
            continue

        # Only unroll forward loops: `i < N` (induction on left, bound on right).
        # Backward loops (i >= N, i > N) have different trip-count semantics
        # and would require the pre-header initial value to unroll correctly.
        # Flipped `N < i` form (lhs=Const) is also supported as `i > N` reversed.
        bound = None
        induction_var = None
        if (cmp_inst.op == CmpOp.LT
                and isinstance(cmp_inst.rhs, Const)
                and isinstance(cmp_inst.lhs, Value)):
            bound = int(cmp_inst.rhs.value)
            induction_var = cmp_inst.lhs
        elif (cmp_inst.op == CmpOp.GT
                and isinstance(cmp_inst.lhs, Const)
                and isinstance(cmp_inst.rhs, Value)):
            # `N > i` is equivalent to `i < N`
            bound = int(cmp_inst.lhs.value)
            induction_var = cmp_inst.rhs

        if bound is None or not isinstance(induction_var, Value):
            continue
        if bound <= 0 or bound > 64:
            continue

        true_label = cond_bb.terminator.true_bb
        false_label = cond_bb.terminator.false_bb
        if true_label not in blocks_by_label or false_label not in blocks_by_label:
            continue

        body_bb = blocks_by_label[true_label]
        exit_bb = blocks_by_label[false_label]

        if not isinstance(getattr(body_bb, 'terminator', None), BrTerm):
            continue
        inc_label = body_bb.terminator.target
        if inc_label not in blocks_by_label:
            continue
        inc_bb = blocks_by_label[inc_label]

        if not isinstance(getattr(inc_bb, 'terminator', None), BrTerm):
            continue
        if inc_bb.terminator.target != cond_bb.label:
            continue

        # Find loop-carried variables from both the body and increment blocks.
        # Pattern: add CANONICAL, NEW_VAL, 0  (writeback copy)
        # The CANONICAL is the loop-carried var, NEW_VAL is the body's output.
        # Body-modified variables (e.g. accumulators like `sum`) write back inside
        # the body block; induction variables write back in the increment block.
        # Scan body_bb first so inc_bb can overwrite if the same var appears in both.
        carried_vars = {}  # canonical_id → (canonical_Value, new_Value)
        for check_bb in [body_bb, inc_bb]:
            for inst in check_bb.instructions:
                if isinstance(inst, BinInst) and inst.op == BinOp.ADD:
                    if isinstance(inst.rhs, Const) and inst.rhs.value == 0:
                        # add CANONICAL, NEW_VAL, 0 → writeback
                        if isinstance(inst.lhs, Value):
                            carried_vars[inst.dest.id] = (inst.dest, inst.lhs)
                    elif isinstance(inst.lhs, Const) and inst.lhs.value == 0:
                        if isinstance(inst.rhs, Value):
                            carried_vars[inst.dest.id] = (inst.dest, inst.rhs)

        # Determine the initial value of the induction variable by scanning
        # all blocks outside the loop for the canonical writeback that seeds it.
        # Pattern: BinInst(dest=induction_var, op=ADD, lhs=Const(init), rhs=Const(0))
        # If init != 0, the trip count is bound - init and i = init + iteration.
        loop_labels = {cond_bb.label, body_bb.label, inc_bb.label}
        init_val = 0
        for scan_bb in kernel.blocks:
            if scan_bb.label in loop_labels:
                continue
            for inst in scan_bb.instructions:
                if (isinstance(inst, BinInst) and inst.op == BinOp.ADD
                        and inst.dest.id == induction_var.id):
                    if isinstance(inst.rhs, Const) and int(inst.rhs.value) == 0:
                        if isinstance(inst.lhs, Const):
                            init_val = int(inst.lhs.value)
                    elif isinstance(inst.lhs, Const) and int(inst.lhs.value) == 0:
                        if isinstance(inst.rhs, Const):
                            init_val = int(inst.rhs.value)

        # Detect the induction step from inc_bb.
        # Pattern: BinInst(op=ADD, lhs=induction_var, rhs=Const(step)) → step != 1
        # Default step = 1 (simple i++ loops).
        step = 1
        for inst in inc_bb.instructions:
            if (isinstance(inst, BinInst) and inst.op == BinOp.ADD):
                if (isinstance(inst.lhs, Value) and inst.lhs.id == induction_var.id
                        and isinstance(inst.rhs, Const)):
                    s = int(inst.rhs.value)
                    if s > 0:
                        step = s
                elif (isinstance(inst.rhs, Value) and inst.rhs.id == induction_var.id
                        and isinstance(inst.lhs, Const)):
                    s = int(inst.lhs.value)
                    if s > 0:
                        step = s

        span = bound - init_val
        if span <= 0:
            continue
        # trip_count = ceil(span / step) — only unroll if it divides evenly
        if span % step != 0:
            continue
        trip_count = span // step
        if trip_count > 64:
            continue

        loops.append({
            'cond_bb': cond_bb,
            'body_bb': body_bb,
            'inc_bb': inc_bb,
            'exit_bb': exit_bb,
            'bound': bound,
            'init_val': init_val,
            'step': step,
            'trip_count': trip_count,
            'induction_var': induction_var,
            'carried_vars': carried_vars,
        })

    return loops


def unroll_loops(kernel: Kernel, max_unroll: int = 16) -> int:
    """Unroll eligible for-loops with loop-carried variable chaining."""
    loops = _find_unrollable_loops(kernel)
    unrolled_count = 0

    for loop in loops:
        trip_count = loop['trip_count']
        if trip_count > max_unroll:
            continue

        cond_bb = loop['cond_bb']
        body_bb = loop['body_bb']
        inc_bb = loop['inc_bb']
        exit_bb = loop['exit_bb']
        induction_var = loop['induction_var']
        carried_vars = loop['carried_vars']
        init_val = loop['init_val']
        step = loop['step']

        # Build the value mapping for each iteration.
        # Start: induction_var → Const(init_val), carried vars → their canonical Values
        # Each iteration: create new Values, chain carried vars from prev output.

        all_unrolled_insts = []

        # Persistent remap across iterations for loop-carried variables
        carried_remap = {}  # canonical_id → current Value (chains across iterations)

        for iteration in range(trip_count):
            # Build replacement map: start from carried state + induction var
            remap = dict(carried_remap)  # inherit carried var chain
            # i_value = init_val + step * iteration
            remap[induction_var.id] = Const(induction_var.ty, init_val + step * iteration)

            # Carried variables: for iteration 0, use the canonical (entry) value.
            # For iteration N>0, use the output from iteration N-1.
            # (The previous iteration's "new_val" becomes this iteration's input.)
            # We'll update carried_var mapping after processing each iteration.

            # Copy body instructions with remapping
            iter_new_vals = {}  # Maps body dest id → new Value for this iteration

            for inst in body_bb.instructions:
                new_inst = _remap_inst(inst, remap, kernel, iteration)
                if new_inst is not None:
                    all_unrolled_insts.append(new_inst)
                    # Track the new dest for carried variable chaining
                    if (hasattr(new_inst, 'dest') and hasattr(inst, 'dest')
                            and inst.dest is not None and new_inst.dest is not None):
                        iter_new_vals[inst.dest.id] = new_inst.dest

            # After processing body: update carried variable mapping for NEXT iteration.
            # Use the canonical's OWN writeback result (iter_new_vals[canonical_id]),
            # NOT iter_new_vals[new_val.id].  The difference matters when new_val is
            # itself a carried variable that got updated in this same iteration —
            # chasing new_val would give the post-update value instead of the pre-update
            # value that the writeback correctly captured.  Example: Fibonacci
            # `a = b; b = tmp` — a's writeback captures old-b; if we looked up b's
            # iter_new_val we'd get new-b (= a+b), doubling instead of stepping.
            for canonical_id, (canonical_val, new_val) in carried_vars.items():
                if canonical_id in iter_new_vals:
                    carried_remap[canonical_id] = iter_new_vals[canonical_id]

        # After all iterations: write back carried variables to canonical registers
        for canonical_id, (canonical_val, new_val) in carried_vars.items():
            if canonical_id in carried_remap and isinstance(carried_remap[canonical_id], Value):
                final_val = carried_remap[canonical_id]
                # Emit: canonical = final_val + 0 (writeback copy)
                zero = Const(canonical_val.ty, 0.0 if hasattr(canonical_val.ty, 'is_float') and canonical_val.ty.is_float else 0)
                all_unrolled_insts.append(
                    BinInst(canonical_val, BinOp.ADD, final_val, zero))

        # Replace cond block with unrolled instructions → branch to exit
        cond_bb.instructions = all_unrolled_insts
        cond_bb.terminator = BrTerm(exit_bb.label)

        # Remove dead body and inc blocks
        kernel.blocks = [bb for bb in kernel.blocks
                         if bb.label not in (body_bb.label, inc_bb.label)]

        unrolled_count += 1

    return unrolled_count


def _remap_inst(inst, remap: dict, kernel: Kernel, iteration: int):
    """Copy an instruction, replacing Values according to remap."""

    def _r(op: Operand) -> Operand:
        if isinstance(op, Value) and op.id in remap:
            return remap[op.id]
        return op

    if isinstance(inst, BinInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = BinInst(new_dest, inst.op, _r(inst.lhs), _r(inst.rhs))
        # Update remap so later instructions in this iteration see the new dest
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, LoadInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = LoadInst(new_dest, _r(inst.addr))
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, StoreInst):
        return StoreInst(_r(inst.addr), _r(inst.value))

    elif isinstance(inst, CmpInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = CmpInst(new_dest, inst.op, _r(inst.lhs), _r(inst.rhs))
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, CvtInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = CvtInst(new_dest, _r(inst.src))
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, CallInst):
        if inst.func == '__syncthreads':
            return inst  # keep barriers
        # Remap args for other CallInsts (atomics, warp shuffles, etc.)
        new_args = [_r(a) for a in inst.args]
        if inst.dest is not None:
            new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
            remap[inst.dest.id] = new_dest
            return CallInst(new_dest, inst.func, new_args)
        return CallInst(None, inst.func, new_args)

    elif isinstance(inst, PrintfInst):
        # Remap Value args so each unrolled iteration uses the correct registers.
        new_args = [_r(a) for a in inst.args]
        return PrintfInst(inst.fmt, new_args)

    return inst
