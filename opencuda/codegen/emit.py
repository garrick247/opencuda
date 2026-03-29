"""
OpenCUDA codegen — lower IR to PTX text, then compile via OpenPTXas.

Strategy: IR → PTX text → OpenPTXas pipeline → cubin.
This reuses OpenPTXas's full backend (parser, regalloc, isel, scoreboard, emitter).
"""

from __future__ import annotations
import struct
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         SymbolRef, GlobalAddrInst,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, PrintfInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)
from ..ir.types import (Type, ScalarTy, PtrTy, StructTy, ScalarType, AddrSpace,
                         INT32, UINT32, INT64, UINT64, FLOAT, VOID, DOUBLE, HALF)


def _ptx_type(ty: Type) -> str:
    """Convert IR type to PTX type string."""
    if isinstance(ty, PtrTy):
        return 'u64'
    if isinstance(ty, ScalarTy):
        mapping = {
            ScalarType.VOID: 'u32',
            ScalarType.INT8:  's8',  ScalarType.UINT8:  'u8',
            ScalarType.INT16: 's16', ScalarType.UINT16: 'u16',
            ScalarType.INT32: 's32', ScalarType.UINT32: 'u32',
            ScalarType.INT64: 's64', ScalarType.UINT64: 'u64',
            ScalarType.FLOAT: 'f32', ScalarType.DOUBLE: 'f64',
            ScalarType.HALF: 'f16',
        }
        return mapping.get(ty.scalar, 'u32')
    return 'u32'


def _ptx_reg_prefix(ty: Type) -> str:
    """Get PTX register prefix for a type."""
    if isinstance(ty, PtrTy):
        return 'rd'
    if isinstance(ty, ScalarTy):
        if ty.scalar == ScalarType.HALF:
            return 'h'
        if ty.scalar == ScalarType.DOUBLE:
            return 'fd'
        if ty.size == 8:
            return 'rd'
        if ty.is_float:
            return 'f'
        return 'r'
    return 'r'


def _is_ptr(ty: Type) -> bool:
    return isinstance(ty, PtrTy)


def _is_float(ty: Type) -> bool:
    return isinstance(ty, ScalarTy) and ty.is_float


def _is_half(ty: Type) -> bool:
    return isinstance(ty, ScalarTy) and ty.scalar == ScalarType.HALF


def _is_64bit(ty: Type) -> bool:
    if isinstance(ty, PtrTy):
        return True
    if isinstance(ty, ScalarTy):
        return ty.size == 8
    return False


def _half_hex(f: float) -> str:
    """Convert float to IEEE 754 half-precision hex string."""
    return struct.pack('>e', f).hex().upper()


def _cvt_modifier(dst_ty: Type, src_ty: Type) -> str:
    """Return the rounding modifier needed for a cvt instruction, or ''."""
    dst_is_float = isinstance(dst_ty, ScalarTy) and dst_ty.is_float
    src_is_float = isinstance(src_ty, ScalarTy) and src_ty.is_float
    dst_is_int = isinstance(dst_ty, ScalarTy) and not dst_ty.is_float
    src_is_int = isinstance(src_ty, ScalarTy) and not src_ty.is_float

    if src_is_float and dst_is_int:
        return '.rzi'  # float → int: round toward zero
    if src_is_int and dst_is_float:
        return '.rn'   # int → float: round to nearest
    if src_is_float and dst_is_float:
        # float → float: rounding only needed when narrowing
        src_size = src_ty.size if isinstance(src_ty, ScalarTy) else 4
        dst_size = dst_ty.size if isinstance(dst_ty, ScalarTy) else 4
        if dst_size < src_size:
            return '.rn'
    return ''


def _build_alloc_map(kernel: Kernel):
    """Linear scan register allocator.

    Returns (alloc, val_type_map, pred_ids, alloc_max) where:
      alloc: {val_id: phys_index}
      val_type_map: {val_id: Type}
      pred_ids: set of val_ids that are predicates (CmpInst destinations)
      alloc_max: {prefix: max_phys_index_used}
    """
    val_type_map = {}
    pred_ids = set()
    live_start = {}
    live_end = {}

    # Flatten all instructions with a global index
    flat = []
    for bb in kernel.blocks:
        for inst in bb.instructions:
            flat.append(inst)
        if bb.terminator:
            flat.append(bb.terminator)

    def _note_def(val, idx):
        if isinstance(val, Value):
            val_type_map[val.id] = val.ty
            if val.id not in live_start:
                live_start[val.id] = idx
            live_end[val.id] = idx

    def _note_use(op, idx):
        if isinstance(op, Value):
            val_type_map.setdefault(op.id, op.ty)
            # Extend live_start backwards: if an inline merge block is created
            # before its inline body blocks in kernel.blocks, the use appears at
            # a lower flat index than the def.  live_start must cover that use.
            live_start[op.id] = min(live_start.get(op.id, idx), idx)
            live_end[op.id] = max(live_end.get(op.id, idx), idx)

    # Register local spill pointer Values as defined at position 0 so their
    # live range starts at the beginning of the function (preamble), not at
    # their first in-block use. Without this, the allocator may reuse their
    # register for short-lived Values defined before the first in-block use.
    if hasattr(kernel, '_local_decls'):
        for _lname, _lty, _lcount, lval in kernel._local_decls:
            _note_def(lval, 0)

    # Register shared memory base-address Values at position 0 for the same
    # reason: their mov.u64 initializer is emitted before entry_1, so any
    # Value whose live range starts later (from its first in-block use) could
    # be assigned the same physical register, corrupting the smem address.
    if hasattr(kernel, '_shared_decls'):
        _smem_names = {s[0] for s in kernel._shared_decls}
        for bb in kernel.blocks:
            for inst in bb.instructions:
                for _attr in ('lhs', 'dest', 'addr'):
                    _v = getattr(inst, _attr, None)
                    if (isinstance(_v, Value)
                            and _v.name in _smem_names
                            and isinstance(_v.ty, PtrTy)):
                        _note_def(_v, 0)

    for i, inst in enumerate(flat):
        if isinstance(inst, (BinInst, CmpInst, CvtInst)):
            _note_def(inst.dest, i)
            if isinstance(inst, CmpInst):
                pred_ids.add(inst.dest.id)
        # Propagate predicate-ness through AND/OR/XOR of predicates.
        # This handles `&&` / `||`: if both operands are predicates, the result is too.
        # Runs as a fixpoint because chains are rare in practice.
        if (isinstance(inst, BinInst)
                and inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR)
                and isinstance(inst.lhs, Value) and inst.lhs.id in pred_ids
                and isinstance(inst.rhs, Value) and inst.rhs.id in pred_ids):
            pred_ids.add(inst.dest.id)
        if isinstance(inst, (LoadInst,)):
            _note_def(inst.dest, i)
        if isinstance(inst, (CallInst,)):
            if inst.dest:
                _note_def(inst.dest, i)
        if isinstance(inst, (ParamInst,)):
            _note_def(inst.dest, i)
        if isinstance(inst, GlobalAddrInst):
            _note_def(inst.dest, i)

        # Uses
        if isinstance(inst, BinInst):
            _note_use(inst.lhs, i)
            _note_use(inst.rhs, i)
        elif isinstance(inst, CmpInst):
            _note_use(inst.lhs, i)
            _note_use(inst.rhs, i)
        elif isinstance(inst, LoadInst):
            _note_use(inst.addr, i)
        elif isinstance(inst, StoreInst):
            _note_use(inst.addr, i)
            _note_use(inst.value, i)
        elif isinstance(inst, CvtInst):
            _note_use(inst.src, i)
        elif isinstance(inst, CallInst):
            for a in inst.args:
                _note_use(a, i)
        elif isinstance(inst, PrintfInst):
            for a in inst.args:
                _note_use(a, i)
        elif isinstance(inst, CondBrTerm):
            _note_use(inst.cond, i)

    # Back-edge liveness extension: values defined before a loop header and used
    # within the loop body must remain live until the loop's back edge, otherwise
    # the flat-order allocator incorrectly reuses their register after their last
    # flat use but before the back edge (which re-enters the loop body).
    #
    # Algorithm:
    #   1. Build flat-start index for each basic block.
    #   2. Detect back edges: a branch whose target block starts at or before the
    #      source block's own start (i.e., the branch goes backwards in flat order).
    #   3. For every back edge from source_end→header_start, extend the live_end
    #      of any value V where live_start[V] <= header_start and
    #      header_start <= live_end[V] <= source_end  (V is defined before or at
    #      the header and last-used inside the loop body).
    bb_flat_start: dict[str, int] = {}
    bb_flat_end: dict[str, int] = {}
    _pos = 0
    for bb in kernel.blocks:
        bb_flat_start[bb.label] = _pos
        n_insts = len(bb.instructions) + (1 if bb.terminator else 0)
        bb_flat_end[bb.label] = _pos + n_insts - 1
        _pos += n_insts

    back_edges: list[tuple[int, int, int]] = []  # (header_start, header_end, latch_end)
    for bb in kernel.blocks:
        if bb.terminator is None:
            continue
        src_start = bb_flat_start[bb.label]
        targets: list[str] = []
        if isinstance(bb.terminator, BrTerm):
            targets = [bb.terminator.target]
        elif isinstance(bb.terminator, CondBrTerm):
            targets = [bb.terminator.true_bb, bb.terminator.false_bb]
        for tgt in targets:
            if tgt in bb_flat_start and bb_flat_start[tgt] <= src_start:
                back_edges.append((bb_flat_start[tgt], bb_flat_end[tgt], bb_flat_end[bb.label]))

    for header_start, header_end, loop_end in back_edges:
        for val_id, le in list(live_end.items()):
            ls = live_start.get(val_id, 0)
            if not (header_start <= le <= loop_end):
                continue
            # Classic case: value defined before the loop header.
            classic = ls <= header_start
            # Inline-merge case: value *used* within the header block (ls is
            # inside [header_start, header_end]) but *defined* in the loop body
            # (le > header_end).  This arises when an inline merge block is placed
            # before the ternary/branch blocks that write its inputs in the flat
            # instruction list, so the use appears at a lower flat index than the
            # definition even though execution visits the definition first.
            inline_merge = (ls <= header_end and le > header_end)
            if classic or inline_merge:
                live_end[val_id] = loop_end

    # Group vals by prefix bucket
    buckets = {}
    for val_id, ty in val_type_map.items():
        if val_id in pred_ids:
            prefix = 'p'
        else:
            prefix = _ptx_reg_prefix(ty)
        start = live_start.get(val_id, 0)
        end = live_end.get(val_id, start)
        buckets.setdefault(prefix, []).append((start, end, val_id))

    # Linear scan per bucket
    alloc = {}
    alloc_max = {}

    for prefix, intervals in buckets.items():
        intervals.sort(key=lambda x: x[0])
        free_list = []
        next_reg = 0
        active = []  # (end, phys_idx, val_id)

        for start, end, val_id in intervals:
            # Expire old intervals
            new_active = []
            for a_end, a_phys, a_id in active:
                if a_end < start:
                    free_list.append(a_phys)
                else:
                    new_active.append((a_end, a_phys, a_id))
            active = new_active

            # Assign register
            if free_list:
                phys = free_list.pop(0)
            else:
                phys = next_reg
                next_reg += 1

            alloc[val_id] = phys
            active.append((end, phys, val_id))
            alloc_max[prefix] = max(alloc_max.get(prefix, 0), phys + 1)

    return alloc, val_type_map, pred_ids, alloc_max


class PTXEmitter:
    """Emits PTX text from IR."""

    def __init__(self):
        self._lines: list[str] = []
        self._reg_counts: dict[str, int] = {}
        self._module_preamble: list[str] = []
        self._emitted_globals: set[str] = set()
        self._printf_strings: dict[str, str] = {}
        self._printf_call_count: int = 0
        # Linear scan allocation results (set per kernel in emit_kernel)
        self._alloc: dict[int, int] = {}
        self._alloc_max: dict[str, int] = {}
        self._fallback_alloc: dict[int, int] = {}   # emission-time values
        self._fallback_count: dict[str, int] = {}   # per-prefix next index
        self._pred_ids: set[int] = set()
        self._val_type_map: dict[int, Type] = {}

    def _alloc_pred(self) -> str:
        """Allocate a scratch predicate register for emission-time temporaries."""
        base = self._alloc_max.get('p', 0)
        idx = base + self._fallback_count.get('p', 0)
        self._fallback_count['p'] = self._fallback_count.get('p', 0) + 1
        self._reg_counts['p'] = max(self._reg_counts.get('p', 0), idx + 1)
        return f'%p{idx}'

    def _reg(self, v: Value) -> str:
        prefix = 'p' if v.id in self._pred_ids else _ptx_reg_prefix(v.ty)
        if v.id in self._alloc:
            phys = self._alloc[v.id]
        elif v.id in self._fallback_alloc:
            phys = self._fallback_alloc[v.id]
        else:
            # Emission-time value (widen cache, printf temps, etc.) — assign
            # the next compact index above the linear-scan allocated range.
            base = self._alloc_max.get(prefix, 0)  # alloc_max stores count (max+1)
            phys = base + self._fallback_count.get(prefix, 0)
            self._fallback_count[prefix] = self._fallback_count.get(prefix, 0) + 1
            self._fallback_alloc[v.id] = phys
        self._reg_counts[prefix] = max(self._reg_counts.get(prefix, 0), phys + 1)
        return f'%{prefix}{phys}'

    def _operand(self, op: Operand, force_type: str = None, kernel=None) -> str:
        if isinstance(op, Value):
            # If this value is a predicate register being used in an integer context,
            # emit selp to convert pred → 0/1 integer.  Predicates cannot be used
            # directly as operands to add/mul/cvt/st — only to @pred bra and setp.
            if (kernel is not None
                    and op.id in self._pred_ids
                    and force_type not in (None, 'pred')):
                return self._pred_to_int(op, force_type if force_type else 's32', kernel)
            return self._reg(op)
        if isinstance(op, SymbolRef):
            # Symbol reference: the symbol name serves as a generic address.
            # Used in [sym_name] address context for ld/st/atom instructions.
            return op.sym_name
        if isinstance(op, Const):
            is_f64 = (force_type == 'f64') or (
                force_type is None and isinstance(op.ty, ScalarTy)
                and op.ty.is_float and op.ty.size == 8)
            is_fp = force_type in ('f32', 'f64') if force_type else (
                isinstance(op.ty, ScalarTy) and op.ty.is_float)
            is_half = force_type == 'f16'
            if is_half:
                return f'0h{_half_hex(float(op.value))}'
            if is_f64:
                # Use 0d (64-bit IEEE double) literal to preserve double precision.
                # 0f literals are 32-bit and lose precision for non-exact values.
                return f'0d{self._double_hex(float(op.value))}'
            if is_fp:
                return f'0f{self._float_hex(float(op.value))}'
            return str(int(op.value))
        return str(op)

    def _float_hex(self, f: float) -> str:
        return struct.pack('>f', f).hex().upper()

    def _double_hex(self, f: float) -> str:
        return struct.pack('>d', f).hex().upper()

    def _coerce_to_float(self, op: Operand, target_fty: str, kernel: Kernel) -> str:
        """Return a PTX operand string that is guaranteed to be of target_fty type.

        Emits a conversion instruction if the operand's type does not match
        the target float type (e.g., int→f32, half→f32, half→f64).
        Const operands are returned directly as float literals.
        """
        if isinstance(op, Const):
            return self._operand(op, target_fty)
        if not isinstance(op, Value):
            return str(op)
        op_ty = op.ty
        op_ptx = _ptx_type(op_ty) if isinstance(op_ty, ScalarTy) else None
        if op_ptx == target_fty:
            return self._reg(op)
        # Emit a conversion instruction
        from ..ir.types import FLOAT, DOUBLE, HALF, INT32, UINT32
        dest_ty = {'f32': FLOAT, 'f64': DOUBLE, 'f16': HALF}.get(target_fty, FLOAT)
        tmp = kernel.new_value(f'_coerce_{op.id}', dest_ty)
        if op_ptx == 'f16' and target_fty == 'f32':
            self._lines.append(f'    cvt.f32.f16 {self._reg(tmp)}, {self._reg(op)};')
        elif op_ptx == 'f16' and target_fty == 'f64':
            self._lines.append(f'    cvt.f64.f16 {self._reg(tmp)}, {self._reg(op)};')
        elif op_ptx in ('s32', 'u32') and target_fty == 'f32':
            self._lines.append(f'    cvt.rn.f32.{op_ptx} {self._reg(tmp)}, {self._reg(op)};')
        elif op_ptx in ('s32', 'u32') and target_fty == 'f64':
            self._lines.append(f'    cvt.rn.f64.{op_ptx} {self._reg(tmp)}, {self._reg(op)};')
        elif op_ptx == 'f32' and target_fty == 'f64':
            self._lines.append(f'    cvt.f64.f32 {self._reg(tmp)}, {self._reg(op)};')
        else:
            # Fallback: trust the caller (may produce invalid PTX for unusual combos)
            return self._reg(op)
        return self._reg(tmp)

    def _pred_to_int(self, op: 'Value', dest_ty: str, kernel: 'Kernel') -> str:
        """Convert a predicate register to a 0/1 integer using selp.
        PTX predicate registers cannot be used as integer operands directly.
        Returns the operand string for the converted integer value.
        """
        tmp = kernel.new_value(f'_pred_int_{op.id}', INT32)
        self._lines.append(
            f'    selp.{dest_ty} {self._reg(tmp)}, 1, 0, {self._reg(op)};')
        return self._reg(tmp)

    def emit_kernel(self, kernel: Kernel) -> str:
        self._lines = []
        self._reg_counts = {}
        self._shared_val_ids: dict[str, list] = {}
        self._widen_cache = {}
        self._printf_call_count = 0
        self._fallback_alloc = {}
        self._fallback_count = {}

        # Run linear scan allocation
        self._alloc, self._val_type_map, self._pred_ids, self._alloc_max = _build_alloc_map(kernel)

        # Pre-scan: find Values that are shared memory pointer registers.
        # Only PtrTy Values should be collected; scalar dest Values with the same name
        # (from auto-deref LoadInst on __shared__ scalars) must NOT be included,
        # as emitting mov.u64 into a float register is illegal PTX.
        if hasattr(kernel, '_shared_decls'):
            smem_names = {s[0] for s in kernel._shared_decls}
            for bb in kernel.blocks:
                for inst in bb.instructions:
                    if hasattr(inst, 'lhs') and isinstance(inst.lhs, Value):
                        if inst.lhs.name in smem_names and isinstance(inst.lhs.ty, PtrTy):
                            self._shared_val_ids.setdefault(inst.lhs.name, []).append(inst.lhs)
                    if hasattr(inst, 'dest') and isinstance(inst.dest, Value):
                        if inst.dest.name in smem_names and isinstance(inst.dest.ty, PtrTy):
                            self._shared_val_ids.setdefault(inst.dest.name, []).append(inst.dest)
                    # LoadInst/StoreInst carry the pointer in .addr — scan that too
                    if hasattr(inst, 'addr') and isinstance(inst.addr, Value):
                        if inst.addr.name in smem_names and isinstance(inst.addr.ty, PtrTy):
                            self._shared_val_ids.setdefault(inst.addr.name, []).append(inst.addr)

        # Pre-scan for printf format strings
        kernel_printf_strings = []
        for bb in kernel.blocks:
            for inst in bb.instructions:
                if isinstance(inst, PrintfInst):
                    label = f'_fmt_{kernel.name}_{len(kernel_printf_strings)}'
                    kernel_printf_strings.append((label, inst.fmt, len(inst.args)))

        # First pass: collect all register usage
        body_lines = []
        self._lines = body_lines
        _printf_idx = [0]
        for bb in kernel.blocks:
            self._emit_block(bb, kernel, kernel_printf_strings, _printf_idx)

        # Build the full PTX
        ptx = []
        ptx.append('.version 9.0')
        ptx.append('.target sm_120')
        ptx.append('.address_size 64')
        ptx.append('')

        # Kernel signature — half params use .b16 (PTX has no .param .f16)
        def _param_ptx_type(ty):
            t = _ptx_type(ty)
            return 'b16' if t == 'f16' else t

        params = ', '.join(
            f'.param .{_param_ptx_type(p.ty)} {p.name}' for p in kernel.params
        )
        ptx.append(f'.visible .entry {kernel.name}(')
        ptx.append(f'    {params})')
        ptx.append('{')

        # Shared memory declarations
        if hasattr(kernel, '_shared_decls'):
            for sname, sty, scount in kernel._shared_decls:
                ptx_sty = _ptx_type(sty)
                if scount == 0:
                    # extern __shared__ (dynamic): must be declared at module level
                    # as ".extern .shared .align N .b8 name[];" — PTX does not allow
                    # incomplete-type shared variables inside a function body.
                    elem_align = sty.size if isinstance(sty, ScalarTy) else 4
                    decl = f'.extern .shared .align {elem_align} .b8 {sname}[];'
                    if decl not in self._module_preamble:
                        self._module_preamble.append(decl)
                else:
                    if isinstance(sty, StructTy):
                        # Struct elements: declare as raw bytes with natural alignment.
                        # PTX has no struct type; use .align X .b8 name[total_bytes].
                        total_bytes = sty.size * scount
                        struct_align = max(
                            (ft.size for _, ft in sty.fields if hasattr(ft, 'size')),
                            default=4)
                        # Clamp to PTX valid alignment (power of 2, max 16)
                        sa = 1
                        while sa < struct_align and sa < 16:
                            sa <<= 1
                        ptx.append(
                            f'    .shared .align {sa} .b8 {sname}[{total_bytes}];')
                    else:
                        ptx.append(f'    .shared .{ptx_sty} {sname}[{scount}];')

        # Local memory (stack) array declarations.
        # Use value ID in the PTX symbol to guarantee uniqueness — two inlined
        # device functions may both declare a local array with the same C name.
        if hasattr(kernel, '_local_decls'):
            for _lname, lty, lcount, lval in kernel._local_decls:
                elem_bytes = lty.size if hasattr(lty, 'size') else 4
                total_bytes = elem_bytes * lcount
                # PTX .align must be a power of two: round elem_bytes up
                align = 1
                while align < elem_bytes:
                    align <<= 1
                align = min(align, 16)  # PTX max useful alignment
                sym = f'_local_{lval.id}'
                ptx.append(f'    .local .align {align} .b8 {sym}[{total_bytes}];')

        # Register declarations
        for prefix, count in sorted(self._reg_counts.items()):
            if prefix == 'rd':
                ptx.append(f'    .reg .b64 %{prefix}<{count}>;')
            elif prefix == 'f':
                ptx.append(f'    .reg .f32 %{prefix}<{count}>;')
            elif prefix == 'fd':
                ptx.append(f'    .reg .f64 %{prefix}<{count}>;')
            elif prefix == 'h':
                ptx.append(f'    .reg .f16 %{prefix}<{count}>;')
            elif prefix == 'p':
                pass  # handled below
            else:
                ptx.append(f'    .reg .b32 %{prefix}<{count}>;')
        pred_count = max(self._reg_counts.get('p', 1), 1)
        ptx.append(f'    .reg .pred %p<{pred_count}>;')
        ptx.append('')

        # Initialize local (stack) array base pointers.
        if hasattr(kernel, '_local_decls'):
            local_inits = []
            emitted_local_regs: set[str] = set()
            for _lname, _lty, _lcount, lval in kernel._local_decls:
                reg = self._reg(lval)
                if reg not in emitted_local_regs:
                    emitted_local_regs.add(reg)
                    sym = f'_local_{lval.id}'
                    local_inits.append(f'    mov.u64 {reg}, {sym};')
            body_lines = local_inits + body_lines

        # Initialize shared memory base addresses — one mov per unique phys register.
        # Multiple Values can alias to the same physical register after linear-scan
        # allocation; only emit one initializer per register name to avoid redundant
        # `mov.u64 %rd1, smem; mov.u64 %rd1, smem; ...` sequences.
        if hasattr(kernel, '_shared_decls'):
            smem_inits = []
            emitted_smem_regs: set[str] = set()
            for sname, sty, scount in kernel._shared_decls:
                for val in self._shared_val_ids.get(sname, []):
                    reg = self._reg(val)
                    if reg not in emitted_smem_regs:
                        emitted_smem_regs.add(reg)
                        smem_inits.append(f'    mov.u64 {reg}, {sname};')
            body_lines = smem_inits + body_lines

        # Body
        ptx.extend(body_lines)

        ptx.append('}')

        # Build preamble for printf strings (module-level globals)
        preamble_lines = []
        if kernel_printf_strings:
            if '__vprintf_declared' not in self._emitted_globals:
                self._emitted_globals.add('__vprintf_declared')
                preamble_lines.append('.extern .func (.param .b32 func_retval0) vprintf(.param .b64 fmt, .param .b64 valist);')
                preamble_lines.append('')
            for label, fmt, nargs in kernel_printf_strings:
                if label not in self._emitted_globals:
                    self._emitted_globals.add(label)
                    encoded = fmt.encode('utf-8') + b'\x00'
                    byte_list = ', '.join(str(b) for b in encoded)
                    preamble_lines.append(f'.global .align 1 .b8 {label}[{len(encoded)}] = {{{byte_list}}};')

        if preamble_lines:
            self._module_preamble.extend(preamble_lines)

        return '\n'.join(ptx)

    def _emit_block(self, bb: BasicBlock, kernel: Kernel,
                    printf_strings=None, printf_idx=None):
        if bb.label != 'entry':
            self._lines.append(f'{bb.label}:')

        for inst in bb.instructions:
            self._emit_inst(inst, kernel, printf_strings, printf_idx)

        if bb.terminator:
            self._emit_term(bb.terminator)

    def _emit_inst(self, inst, kernel: Kernel,
                   printf_strings=None, printf_idx=None):
        if isinstance(inst, GlobalAddrInst):
            dest = self._reg(inst.dest)
            # cvta.to.{space} requires a register source, not a symbol name.
            # Use mov.u64 to load the generic address, then convert to state space.
            self._lines.append(f'    mov.u64 {dest}, {inst.sym_name};')
            if inst.addr_space == AddrSpace.CONST:
                self._lines.append(f'    cvta.to.const.u64 {dest}, {dest};')
            else:
                self._lines.append(f'    cvta.to.global.u64 {dest}, {dest};')
            return
        if isinstance(inst, ParamInst):
            ty = kernel.params[inst.param_index].ty
            ptx_ty = _ptx_type(ty)
            # PTX does not support ld.param.f16 — use b16 instead
            if ptx_ty == 'f16':
                ptx_ty = 'b16'
            self._lines.append(
                f'    ld.param.{ptx_ty} {self._reg(inst.dest)}, [{inst.param_name}];')

        elif isinstance(inst, BinInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            # Sub-word arithmetic: PTX has no s8/u8/s16/u16 ALU instructions.
            # Promote to 32-bit for arithmetic (loads/stores use the actual type).
            if ptx_ty in ('s8', 'u8', 's16', 'u16'):
                ptx_ty = 's32' if ptx_ty in ('s8', 's16') else 'u32'

            # Emit mov for add-zero copy patterns (loop writeback / initializers).
            # add D, V, 0  or  add D, 0, V  → mov D, V
            # add D, 0, 0              → mov D, 0
            # This avoids wasteful ALU slots for pure register copies.
            if (inst.op == BinOp.ADD
                    and not _is_ptr(ty)):         # pointer add has widen logic
                lhs_zero = isinstance(inst.lhs, Const) and inst.lhs.value == 0
                rhs_zero = isinstance(inst.rhs, Const) and inst.rhs.value == 0
                if lhs_zero and rhs_zero:
                    # mov D, 0
                    if _is_half(ty):
                        self._lines.append(f'    cvt.rn.f16.f32 {self._reg(inst.dest)}, 0f00000000;')
                    elif _is_float(ty):
                        fty = _ptx_type(ty)
                        self._lines.append(f'    mov.{fty} {self._reg(inst.dest)}, 0f00000000;')
                    else:
                        self._lines.append(f'    mov.{ptx_ty} {self._reg(inst.dest)}, 0;')
                    return
                elif rhs_zero and _is_half(ty) and isinstance(inst.lhs, Value):
                    # mov.b16 D, V  (add D, V, 0 — half writeback copy; f16 regs are b16)
                    self._lines.append(f'    mov.b16 {self._reg(inst.dest)}, {self._reg(inst.lhs)};')
                    return
                elif lhs_zero and _is_half(ty) and isinstance(inst.rhs, Value):
                    # mov.b16 D, V  (add D, 0, V — half writeback copy)
                    self._lines.append(f'    mov.b16 {self._reg(inst.dest)}, {self._reg(inst.rhs)};')
                    return
                elif rhs_zero and not _is_half(ty):
                    # mov D, V  or  mov D, C  (add D, V/C, 0)
                    if _is_float(ty):
                        fty = _ptx_type(ty)
                        # Always use _coerce_to_float — handles int→float type mismatches.
                        # e.g. ternary(int_arm, float_arm): dest=f32 but lhs=s32 Value
                        src = self._coerce_to_float(inst.lhs, fty, kernel)
                        self._lines.append(f'    mov.{fty} {self._reg(inst.dest)}, {src};')
                    else:
                        self._lines.append(f'    mov.{ptx_ty} {self._reg(inst.dest)}, {self._operand(inst.lhs, ptx_ty, kernel)};')
                    return
                elif lhs_zero and not isinstance(inst.rhs, Const) and not _is_half(ty):
                    # mov D, V  (add D, 0, V)
                    if _is_float(ty):
                        fty = _ptx_type(ty)
                        # Always use _coerce_to_float — handles int→float type mismatches.
                        src = self._coerce_to_float(inst.rhs, fty, kernel)
                        self._lines.append(f'    mov.{fty} {self._reg(inst.dest)}, {src};')
                    else:
                        self._lines.append(f'    mov.{ptx_ty} {self._reg(inst.dest)}, {self._operand(inst.rhs, ptx_ty, kernel)};')
                    return

            op_map = {
                BinOp.ADD: 'add', BinOp.SUB: 'sub', BinOp.MUL: 'mul.lo',
                BinOp.DIV: 'div', BinOp.MOD: 'rem',
                BinOp.AND: 'and', BinOp.OR: 'or', BinOp.XOR: 'xor',
                BinOp.SHL: 'shl', BinOp.SHR: 'shr',
            }
            ptx_op = op_map.get(inst.op, 'add')
            # Float mul doesn't need .lo qualifier
            if inst.op == BinOp.MUL and _is_float(ty):
                ptx_op = 'mul'
            # Float division: PTX requires a modifier.
            # f32: div.approx.f32 (fast, ~22-bit relative error) — matches CUDA default.
            # f64: div.rn.f64 (IEEE round-to-nearest) — rounding modifier mandatory for f64.
            if inst.op == BinOp.DIV and _is_float(ty):
                fty = _ptx_type(ty)
                ptx_op = 'div.approx' if fty == 'f32' else 'div.rn'
            # Half precision arithmetic
            if _is_half(ty):
                half_op_map = {
                    BinOp.ADD: 'add', BinOp.SUB: 'sub',
                    BinOp.MUL: 'mul', BinOp.DIV: 'div.approx',
                }
                ptx_op = half_op_map.get(inst.op, 'add')
                # PTX does not accept 0h#### immediate constants in f16 arithmetic
                # instructions. Materialize any Const operands into h registers first
                # using cvt.rn.f16.f32 (narrowing from f32, which accepts immediates).
                def _half_operand(op):
                    if isinstance(op, Const):
                        tmp = kernel.new_value('_h_tmp', HALF)
                        f32_hex = self._float_hex(float(op.value))
                        self._lines.append(
                            f'    cvt.rn.f16.f32 {self._reg(tmp)}, 0f{f32_hex};')
                        return self._reg(tmp)
                    return self._operand(op, 'f16')
                lhs_str = _half_operand(inst.lhs)
                rhs_str = _half_operand(inst.rhs)
                self._lines.append(
                    f'    {ptx_op}.f16 {self._reg(inst.dest)}, {lhs_str}, {rhs_str};')
                return
            # Predicate AND/OR/XOR: both operands are predicates → use .pred type
            if inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR):
                lhs_is_pred = isinstance(inst.lhs, Value) and inst.lhs.id in self._pred_ids
                rhs_is_pred = isinstance(inst.rhs, Value) and inst.rhs.id in self._pred_ids
                if lhs_is_pred and rhs_is_pred:
                    ptx_op_map = {BinOp.AND: 'and', BinOp.OR: 'or', BinOp.XOR: 'xor'}
                    self._lines.append(
                        f'    {ptx_op_map[inst.op]}.pred {self._reg(inst.dest)}, '
                        f'{self._reg(inst.lhs)}, {self._reg(inst.rhs)};')
                    return
            # AND/OR/XOR/SHL always use bitwise (.b) type — sign doesn't matter.
            # SHR needs sign-awareness: shr.s32 for INT32 (arithmetic, sign-extending),
            # shr.b32 for UINT32 (logical, zero-filling).
            # A C int value of -4 >> 1 must give -2, not 2147483646.
            # Sub-word note: PTX has no b8/b16 ALU — promote to b32.
            if inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR, BinOp.SHL):
                if isinstance(ty, ScalarTy) and ty.size <= 2:
                    ptx_ty = 'b32'
                else:
                    ptx_ty = f'b{ty.size * 8}' if isinstance(ty, ScalarTy) else 'b32'
            elif inst.op == BinOp.SHR:
                if isinstance(ty, ScalarTy) and ty.is_signed:
                    bits = max(ty.size * 8, 32)  # promote s8/s16 → s32
                    ptx_ty = f's{bits}'
                else:
                    bits = max(ty.size * 8, 32)  # promote u8/u16 → b32
                    ptx_ty = f'b{bits}'

            # Pointer arithmetic: use u64 for add/sub
            if _is_ptr(ty) and inst.op in (BinOp.ADD, BinOp.SUB):
                lhs = self._operand(inst.lhs)
                rhs = self._operand(inst.rhs)
                if isinstance(inst.rhs, Const):
                    # Const offsets can be used directly as 64-bit immediates in
                    # add.u64 — no cvt.u64.u32 needed.
                    rhs = str(int(inst.rhs.value))
                elif isinstance(inst.rhs, Value) and not _is_64bit(inst.rhs.ty):
                    rhs_id = inst.rhs.id
                    if rhs_id in self._widen_cache:
                        wide = self._widen_cache[rhs_id]
                    else:
                        wide = kernel.new_value(f'wide{inst.dest.id}', ty)
                        # Use sign-extending cvt for signed types (INT32 → s64),
                        # zero-extending for unsigned (UINT32 → u64).
                        # Wrong extension turns negative indices into huge positive
                        # addresses, silently corrupting out-of-bounds accesses.
                        rhs_ty = inst.rhs.ty
                        if isinstance(rhs_ty, ScalarTy) and rhs_ty.is_signed:
                            widen_op = 'cvt.s64.s32'
                        else:
                            widen_op = 'cvt.u64.u32'
                        self._lines.append(
                            f'    {widen_op} {self._reg(wide)}, {rhs};')
                        self._widen_cache[rhs_id] = wide
                    rhs = self._reg(wide)
                self._lines.append(
                    f'    {ptx_op}.u64 {self._reg(inst.dest)}, {lhs}, {rhs};')
            elif _is_float(ty):
                fty = _ptx_type(ty)  # f32 or f64
                # Coerce operands to the destination float type; this handles
                # mixed-type expressions like float+int or float+half.
                lhs_str = self._coerce_to_float(inst.lhs, fty, kernel)
                rhs_str = self._coerce_to_float(inst.rhs, fty, kernel)
                self._lines.append(
                    f'    {ptx_op}.{fty} {self._reg(inst.dest)}, {lhs_str}, {rhs_str};')
            elif _is_64bit(ty):
                # Widen any 32-bit Value operands to match the 64-bit instruction type.
                # Without this, mul.lo.s64 %rd, %rd, %r  (32-bit %r) is rejected by ptxas.
                # EXCEPTION: shift amount (rhs of SHL/SHR) must remain 32-bit in PTX —
                # shl.b64 / shr.bXX take a b32 shift count, not a b64 register.
                def _bin64_operand(op, tgt_ptx):
                    if isinstance(op, Value) and not _is_64bit(op.ty):
                        wid = self._widen_cache.get(op.id)
                        if wid is None:
                            is_signed = isinstance(op.ty, ScalarTy) and op.ty.is_signed
                            widen_op = 'cvt.s64.s32' if is_signed else 'cvt.u64.u32'
                            wid = kernel.new_value(f'_wide{inst.dest.id}', ty)
                            self._lines.append(
                                f'    {widen_op} {self._reg(wid)}, {self._reg(op)};')
                            self._widen_cache[op.id] = wid
                        return self._reg(wid)
                    return self._operand(op, tgt_ptx, kernel)
                # For shifts, lhs is the value (needs widening), rhs is the amount (stays 32-bit).
                if inst.op in (BinOp.SHL, BinOp.SHR):
                    lhs_str = _bin64_operand(inst.lhs, ptx_ty)
                    # Shift amount must be b32 in PTX — do NOT widen to 64-bit.
                    # If rhs is already a 64-bit Value (e.g. widened earlier),
                    # we need a narrow: but that shouldn't happen in practice.
                    # Just use _operand with b32 hint.
                    if isinstance(inst.rhs, Value) and _is_64bit(inst.rhs.ty):
                        # Unlikely but handle: truncate back to 32-bit
                        narrow = kernel.new_value(f'_sham{inst.dest.id}', UINT32)
                        self._lines.append(
                            f'    cvt.u32.u64 {self._reg(narrow)}, {self._reg(inst.rhs)};')
                        rhs_str = self._reg(narrow)
                    else:
                        rhs_str = self._operand(inst.rhs, 'b32', kernel)
                    self._lines.append(
                        f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, {lhs_str}, {rhs_str};')
                else:
                    self._lines.append(
                        f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                        f'{_bin64_operand(inst.lhs, ptx_ty)}, '
                        f'{_bin64_operand(inst.rhs, ptx_ty)};')
            else:
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty, kernel)}, '
                    f'{self._operand(inst.rhs, ptx_ty, kernel)};')

        elif isinstance(inst, CmpInst):
            # Special case: logical NOT of a predicate register.
            # CmpInst(dest, EQ, pred_val, Const(0)) where pred_val is a predicate
            # should emit 'not.pred %p_dest, %p_src' — NOT setp.eq.s32 (invalid).
            if (inst.op == CmpOp.EQ
                    and isinstance(inst.lhs, Value)
                    and inst.lhs.id in self._pred_ids
                    and isinstance(inst.rhs, Const) and inst.rhs.value == 0):
                pred = self._reg(inst.dest)
                src = self._reg(inst.lhs)
                self._lines.append(f'    not.pred {pred}, {src};')
                return
            if (inst.op == CmpOp.EQ
                    and isinstance(inst.rhs, Value)
                    and inst.rhs.id in self._pred_ids
                    and isinstance(inst.lhs, Const) and inst.lhs.value == 0):
                pred = self._reg(inst.dest)
                src = self._reg(inst.rhs)
                self._lines.append(f'    not.pred {pred}, {src};')
                return
            # Determine comparison type from BOTH operands (C integer promotion).
            # Bug: using only lhs.ty means:
            #   - Const lhs defaults to INT32 even when rhs is UINT32
            #   - INT32 lhs wins over UINT32 rhs → setp.lt.s32 when u32 is needed
            # C §6.3.1.8: if either operand is unsigned, the signed one converts.
            lhs_ty = inst.lhs.ty if isinstance(inst.lhs, (Value, Const)) else INT32
            rhs_ty = inst.rhs.ty if isinstance(inst.rhs, (Value, Const)) else INT32
            # Float comparison: use the float type (wider wins)
            lhs_is_float = isinstance(lhs_ty, ScalarTy) and lhs_ty.is_float
            rhs_is_float = isinstance(rhs_ty, ScalarTy) and rhs_ty.is_float
            if lhs_is_float or rhs_is_float:
                if lhs_is_float and rhs_is_float:
                    ty = lhs_ty if lhs_ty.size >= rhs_ty.size else rhs_ty
                else:
                    ty = lhs_ty if lhs_is_float else rhs_ty
            # 64-bit wins over 32-bit
            elif isinstance(lhs_ty, ScalarTy) and lhs_ty.size == 8:
                ty = lhs_ty
            elif isinstance(rhs_ty, ScalarTy) and rhs_ty.size == 8:
                ty = rhs_ty
            # Unsigned wins over signed at same width
            elif (isinstance(lhs_ty, ScalarTy) and isinstance(rhs_ty, ScalarTy)
                  and not lhs_ty.is_signed):
                ty = lhs_ty  # lhs is unsigned
            elif (isinstance(lhs_ty, ScalarTy) and isinstance(rhs_ty, ScalarTy)
                  and not rhs_ty.is_signed):
                ty = rhs_ty  # rhs is unsigned, promote to it
            else:
                ty = lhs_ty  # same signedness, use lhs
            ptx_ty = _ptx_type(ty)
            # Sub-word comparison: setp.s8/u8/s16/u16 don't exist in PTX.
            # Promote to 32-bit — values in 32-bit registers, semantics preserved.
            if ptx_ty in ('s8', 'u8', 's16', 'u16'):
                ptx_ty = 's32' if ptx_ty in ('s8', 's16') else 'u32'
            # Float != must use unordered NE (neu) to match C/IEEE 754 semantics:
            # NaN != x is true for all x. Ordered NE (ne) returns false for NaN.
            is_float_ty = isinstance(ty, ScalarTy) and ty.is_float
            op_map = {
                CmpOp.LT: 'lt', CmpOp.LE: 'le', CmpOp.GT: 'gt',
                CmpOp.GE: 'ge', CmpOp.EQ: 'eq',
                CmpOp.NE: 'neu' if is_float_ty else 'ne',
            }
            cmp_str = op_map[inst.op]
            pred = self._reg(inst.dest)
            # If the chosen comparison type is 64-bit but an operand is a 32-bit
            # Value, widen it first — PTX setp requires both operands to have the
            # same register width as the type qualifier.
            def _cmp_operand(op, tgt_ty: str) -> str:
                if isinstance(op, Value) and tgt_ty in ('u64', 's64') and not _is_64bit(op.ty):
                    wid = self._widen_cache.get(op.id)
                    if wid is None:
                        is_signed = isinstance(op.ty, ScalarTy) and op.ty.is_signed
                        widen_op = 'cvt.s64.s32' if is_signed else 'cvt.u64.u32'
                        wid = kernel.new_value(f'_cmp_wide', INT64 if is_signed else UINT64)
                        self._lines.append(
                            f'    {widen_op} {self._reg(wid)}, {self._reg(op)};')
                        self._widen_cache[op.id] = wid
                    return self._reg(wid)
                return self._operand(op, tgt_ty)
            self._lines.append(
                f'    setp.{cmp_str}.{ptx_ty} {pred}, '
                f'{_cmp_operand(inst.lhs, ptx_ty)}, {_cmp_operand(inst.rhs, ptx_ty)};')

        elif isinstance(inst, CvtInst):
            # Use Const's declared type when src is a Const — e.g., CvtInst(f32, Const(UINT32, val))
            # should emit cvt.rn.f32.u32, not cvt.rn.f32.s32 (which would sign-extend UINT32 max).
            src_ty = inst.src.ty if isinstance(inst.src, (Value, Const)) else INT32
            dst_ty = inst.dest.ty
            dst_ptx = _ptx_type(dst_ty)
            # Predicate → integer: emit selp.{type} dest, 1, 0, pred
            src_is_bool = (isinstance(src_ty, ScalarTy)
                           and src_ty.scalar == ScalarType.BOOL)
            src_is_pred_val = (isinstance(inst.src, Value)
                               and inst.src.id in self._pred_ids)
            if src_is_bool or src_is_pred_val:
                pred_src = self._operand(inst.src)
                # selp needs type-appropriate literals for true/false values
                if dst_ptx == 'f32':
                    true_lit, false_lit = '0f3F800000', '0f00000000'
                elif dst_ptx == 'f64':
                    true_lit, false_lit = '0d3FF0000000000000', '0d0000000000000000'
                elif dst_ptx == 'f16':
                    true_lit, false_lit = '0h3C00', '0h0000'
                else:
                    true_lit, false_lit = '1', '0'
                self._lines.append(
                    f'    selp.{dst_ptx} {self._reg(inst.dest)}, {true_lit}, {false_lit}, {pred_src};')
                return
            src_ptx = _ptx_type(src_ty)
            rnd = _cvt_modifier(dst_ty, src_ty)
            src_op = self._operand(inst.src, src_ptx, kernel)
            self._lines.append(
                f'    cvt{rnd}.{dst_ptx}.{src_ptx} {self._reg(inst.dest)}, {src_op};')

        elif isinstance(inst, LoadInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            nc = False
            is_volatile = False
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                ptr_ty = inst.addr.ty
                is_volatile = ptr_ty.volatile
                if ptr_ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
                elif ptr_ty.addr_space == AddrSpace.LOCAL:
                    addr_space = 'local'
                elif ptr_ty.addr_space == AddrSpace.CONST:
                    addr_space = 'global'
                    nc = True
            elif isinstance(inst.addr, SymbolRef) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.volatile:
                    is_volatile = True
                if inst.addr.ty.addr_space == AddrSpace.CONST:
                    addr_space = 'const'
                elif inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
            # PTX does not support ld.f16 — use b16 instead
            if ptx_ty == 'f16':
                ptx_ty = 'b16'
            if is_volatile and addr_space in ('global', 'shared'):
                self._lines.append(
                    f'    ld.volatile.{addr_space}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'[{self._operand(inst.addr)}];')
            elif nc:
                self._lines.append(
                    f'    ld.global.nc.{ptx_ty} {self._reg(inst.dest)}, '
                    f'[{self._operand(inst.addr)}];')
            else:
                self._lines.append(
                    f'    ld.{addr_space}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'[{self._operand(inst.addr)}];')

        elif isinstance(inst, StoreInst):
            ty = getattr(inst.value, 'ty', INT32)
            ptx_ty = _ptx_type(ty)
            # PTX does not support st.f16 — use b16 instead
            if ptx_ty == 'f16':
                ptx_ty = 'b16'
            addr_space = 'global'
            is_volatile = False
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                ptr_ty = inst.addr.ty
                is_volatile = ptr_ty.volatile
                if ptr_ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
                elif ptr_ty.addr_space == AddrSpace.LOCAL:
                    addr_space = 'local'
            # Predicate registers cannot be stored directly — convert to 0/1 integer first.
            if isinstance(inst.value, Value) and inst.value.id in self._pred_ids:
                val_str = self._pred_to_int(inst.value, ptx_ty, kernel)
            else:
                val_str = self._operand(inst.value, ptx_ty)
            if is_volatile and addr_space in ('global', 'shared'):
                self._lines.append(
                    f'    st.volatile.{addr_space}.{ptx_ty} [{self._operand(inst.addr)}], {val_str};')
            else:
                self._lines.append(
                    f'    st.{addr_space}.{ptx_ty} [{self._operand(inst.addr)}], {val_str};')

        elif isinstance(inst, PrintfInst):
            # Emit vprintf sequence
            if printf_strings is None:
                return
            idx = printf_idx[0]
            printf_idx[0] += 1
            if idx >= len(printf_strings):
                return
            fmt_label, fmt_str, nargs = printf_strings[idx]
            nargs = len(inst.args)

            n = self._printf_call_count
            self._printf_call_count += 1

            # Get address of format string
            fmt_addr = kernel.new_value(f'_fmt_addr_{n}', PtrTy(VOID, AddrSpace.GLOBAL))
            self._lines.append(f'    mov.u64 {self._reg(fmt_addr)}, {fmt_label};')

            if nargs > 0:
                valist_size = 8 * nargs
                valist_local = kernel.new_value(f'_valist_{n}', PtrTy(VOID, AddrSpace.LOCAL))
                self._lines.append(f'    .local .align 8 .b8 _valist_{n}[{valist_size}];')
                self._lines.append(f'    mov.u64 {self._reg(valist_local)}, _valist_{n};')

                for i, arg in enumerate(inst.args):
                    arg_ty = arg.ty if isinstance(arg, (Value, Const)) else INT32
                    if _is_float(arg_ty) and not _is_64bit(arg_ty):
                        # Promote f32 → f64
                        promoted = kernel.new_value(f'_va_arg_{n}_{i}', DOUBLE)
                        self._lines.append(
                            f'    cvt.f64.f32 {self._reg(promoted)}, {self._operand(arg)};')
                        store_val = self._reg(promoted)
                        store_ty = 'f64'
                    elif not _is_64bit(arg_ty) and not _is_float(arg_ty):
                        # Widen int32 to 64-bit for valist slot.
                        # Use sign extension for signed types (s32 → s64) so that
                        # negative integers print correctly via vprintf %d/%i.
                        widened = kernel.new_value(f'_va_arg_{n}_{i}', PtrTy(VOID, AddrSpace.GLOBAL))
                        if isinstance(arg_ty, ScalarTy) and arg_ty.is_signed:
                            self._lines.append(
                                f'    cvt.s64.s32 {self._reg(widened)}, {self._operand(arg)};')
                        else:
                            self._lines.append(
                                f'    cvt.u64.u32 {self._reg(widened)}, {self._operand(arg)};')
                        store_val = self._reg(widened)
                        store_ty = 'u64'
                    else:
                        store_val = self._operand(arg)
                        store_ty = _ptx_type(arg_ty)

                    if i == 0:
                        slot_addr = self._reg(valist_local)
                    else:
                        offset_val = kernel.new_value(f'_va_off_{n}_{i}', PtrTy(VOID, AddrSpace.LOCAL))
                        self._lines.append(
                            f'    add.u64 {self._reg(offset_val)}, {self._reg(valist_local)}, {i * 8};')
                        slot_addr = self._reg(offset_val)
                    self._lines.append(f'    st.local.{store_ty} [{slot_addr}], {store_val};')

                # Convert local valist ptr to generic
                valist_generic = kernel.new_value(f'_valist_gen_{n}', PtrTy(VOID, AddrSpace.GENERIC))
                self._lines.append(
                    f'    cvta.local.u64 {self._reg(valist_generic)}, {self._reg(valist_local)};')
                valist_ptr = self._reg(valist_generic)
            else:
                null_va = kernel.new_value(f'_valist_null_{n}', PtrTy(VOID, AddrSpace.GLOBAL))
                self._lines.append(f'    mov.u64 {self._reg(null_va)}, 0;')
                valist_ptr = self._reg(null_va)

            # Call vprintf
            rv = kernel.new_value(f'_vprintf_rv_{n}', INT32)
            self._lines.append(f'    {{')
            self._lines.append(f'    .param .b64 _p0_{n};')
            self._lines.append(f'    .param .b64 _p1_{n};')
            self._lines.append(f'    .param .b32 _rv_{n};')
            self._lines.append(f'    st.param.b64 [_p0_{n}], {self._reg(fmt_addr)};')
            self._lines.append(f'    st.param.b64 [_p1_{n}], {valist_ptr};')
            self._lines.append(f'    call.uni (_rv_{n}), vprintf, (_p0_{n}, _p1_{n});')
            self._lines.append(f'    ld.param.b32 {self._reg(rv)}, [_rv_{n}];')
            self._lines.append(f'    }}')

        elif isinstance(inst, CallInst):
            if inst.func.startswith('atomic'):
                atomic_ops = {
                    'atomicAdd': 'add',
                    'atomicMin': 'min', 'atomicMax': 'max',
                    'atomicAnd': 'and', 'atomicOr': 'or', 'atomicXor': 'xor',
                    'atomicExch': 'exch', 'atomicCAS': 'cas',
                    'atomicInc': 'inc', 'atomicDec': 'dec',
                }
                addr = self._operand(inst.args[0]) if inst.args else '%rd0'
                # Detect shared vs global address space for the pointer argument.
                _addr_arg = inst.args[0] if inst.args else None
                _atom_space = 'global'
                if isinstance(_addr_arg, Value) and isinstance(_addr_arg.ty, PtrTy):
                    if _addr_arg.ty.addr_space == AddrSpace.SHARED:
                        _atom_space = 'shared'
                val_ty = 'u32'
                if len(inst.args) > 1:
                    _val_arg = inst.args[1]
                    if isinstance(_val_arg, Value):
                        val_ty = _ptx_type(_val_arg.ty)
                    elif isinstance(_val_arg, Const):
                        # Use the Const's declared type (INT32→s32, UINT32→u32, FLOAT→f32)
                        val_ty = _ptx_type(_val_arg.ty)
                # PTX type constraints per operation:
                # - and/or/xor/exch require b32 (bitwise, no sign semantics)
                # - add/min/max use typed (s32/u32/f32 — sign matters)
                # Override to b32 for bitwise-only operations.
                # atom.add: PTX has no s64 variant — use u64 for 64-bit integer add.
                if inst.func == 'atomicAdd' and val_ty == 's64':
                    val_ty = 'u64'
                if inst.func in ('atomicAnd', 'atomicOr', 'atomicXor', 'atomicExch'):
                    # Keep bit width but strip sign: s32→b32, u32→b32, f32 stays f32
                    if val_ty in ('s32', 'u32'):
                        val_ty = 'b32'
                    elif val_ty in ('s64', 'u64'):
                        val_ty = 'b64'
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                if inst.func == 'atomicSub':
                    # PTX has no atom.sub. Implement as atom.add(-val).
                    # PTX has no neg.u32 either, so emit sub.TYPE tmp, 0, val.
                    val_arg = inst.args[1] if len(inst.args) > 1 else Const(INT32, 0)
                    if isinstance(val_arg, Const):
                        # Negate the constant directly — no extra instruction needed.
                        neg_val_str = str(int(-val_arg.value)) if 'f' not in val_ty \
                            else f'0f{self._float_hex(-float(val_arg.value))}'
                        self._lines.append(
                            f'    atom.{_atom_space}.add.{val_ty} {dest}, [{addr}], {neg_val_str};')
                    else:
                        # Emit: neg_tmp = 0 - val, then atom.add(neg_tmp)
                        neg_tmp = kernel.new_value(f'_neg_{inst.dest.id if inst.dest else "x"}',
                                                   val_arg.ty)
                        zero_str = '0.0' if 'f' in val_ty else '0'
                        self._lines.append(
                            f'    sub.{val_ty} {self._reg(neg_tmp)}, {zero_str}, {self._operand(val_arg, val_ty)};')
                        self._lines.append(
                            f'    atom.{_atom_space}.add.{val_ty} {dest}, [{addr}], {self._reg(neg_tmp)};')
                elif inst.func == 'atomicCAS':
                    # atomicCAS(addr, compare, val) — 3-arg: PTX atom.cas.b32/b64
                    # atom.{space}.cas.bN dest, [addr], compare, val;
                    cmp_arg = inst.args[1] if len(inst.args) > 1 else Const(INT32, 0)
                    new_arg = inst.args[2] if len(inst.args) > 2 else Const(INT32, 0)
                    # Select b32 or b64 based on compare arg type
                    _cas_ty = 'b32'
                    if isinstance(cmp_arg, Value):
                        _ptx = _ptx_type(cmp_arg.ty)
                        if _ptx in ('s64', 'u64', 'b64'):
                            _cas_ty = 'b64'
                    elif isinstance(cmp_arg, Const) and _ptx_type(cmp_arg.ty) in ('s64', 'u64'):
                        _cas_ty = 'b64'
                    self._lines.append(
                        f'    atom.{_atom_space}.cas.{_cas_ty} {dest}, [{addr}], '
                        f'{self._operand(cmp_arg, _cas_ty)}, {self._operand(new_arg, _cas_ty)};')
                else:
                    ptx_op = atomic_ops.get(inst.func, 'add')
                    self._lines.append(
                        f'    atom.{_atom_space}.{ptx_op}.{val_ty} {dest}, [{addr}], {self._operand(inst.args[1], val_ty)};')
            elif inst.func in ('__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync'):
                shfl_map = {
                    '__shfl_sync': 'idx',
                    '__shfl_up_sync': 'up',
                    '__shfl_down_sync': 'down',
                    '__shfl_xor_sync': 'bfly',
                }
                mode = shfl_map[inst.func]
                mask = self._operand(inst.args[0]) if inst.args else '0xFFFFFFFF'
                val = self._operand(inst.args[1]) if len(inst.args) > 1 else '%r0'
                delta = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                # PTX shfl only supports b32; for 64-bit operands split lo/hi.
                val_ty = inst.args[1].ty if len(inst.args) > 1 and isinstance(inst.args[1], Value) else None
                is64 = val_ty is not None and isinstance(val_ty, ScalarTy) and val_ty.size == 8
                if is64 and dest.startswith('%rd'):
                    n = inst.dest.id if inst.dest else 0
                    lo_in  = kernel.new_value(f'_shfl_lo_in_{n}', INT32)
                    hi_in  = kernel.new_value(f'_shfl_hi_in_{n}', INT32)
                    lo_out = kernel.new_value(f'_shfl_lo_out_{n}', INT32)
                    hi_out = kernel.new_value(f'_shfl_hi_out_{n}', INT32)
                    # Unpack 64-bit val into hi/lo 32-bit registers
                    self._lines.append(f'    mov.b64 {{{self._reg(lo_in)}, {self._reg(hi_in)}}}, {val};')
                    self._lines.append(
                        f'    shfl.sync.{mode}.b32 {self._reg(lo_out)}, {self._reg(lo_in)}, {delta}, 31, {mask};')
                    self._lines.append(
                        f'    shfl.sync.{mode}.b32 {self._reg(hi_out)}, {self._reg(hi_in)}, {delta}, 31, {mask};')
                    # Repack into 64-bit dest
                    self._lines.append(f'    mov.b64 {dest}, {{{self._reg(lo_out)}, {self._reg(hi_out)}}};')
                else:
                    self._lines.append(
                        f'    shfl.sync.{mode}.b32 {dest}, {val}, {delta}, 31, {mask};')
            elif inst.func in ('__ballot_sync', '__all_sync', '__any_sync'):
                # PTX vote.sync.* requires a predicate register for the condition arg.
                # If the condition is already a predicate (from CmpInst), use it directly;
                # otherwise emit setp.ne.s32 to convert an integer condition.
                mask = self._operand(inst.args[0]) if inst.args else '0xFFFFFFFF'
                cond_arg = inst.args[1] if len(inst.args) > 1 else None
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                # Check if condition is a predicate Value
                cond_is_pred = (isinstance(cond_arg, Value) and cond_arg.id in self._pred_ids)
                if cond_is_pred:
                    tmp_pred = self._operand(cond_arg)
                else:
                    cond_val = self._operand(cond_arg) if cond_arg is not None else '0'
                    tmp_pred = self._alloc_pred()
                    self._lines.append(f'    setp.ne.s32 {tmp_pred}, {cond_val}, 0;')
                if inst.func == '__ballot_sync':
                    self._lines.append(
                        f'    vote.sync.ballot.b32 {dest}, {tmp_pred}, {mask};')
                elif inst.func == '__all_sync':
                    tmp_pred2 = self._alloc_pred()
                    self._lines.append(
                        f'    vote.sync.all.pred {tmp_pred2}, {tmp_pred}, {mask};')
                    self._lines.append(f'    selp.s32 {dest}, 1, 0, {tmp_pred2};')
                elif inst.func == '__any_sync':
                    tmp_pred2 = self._alloc_pred()
                    self._lines.append(
                        f'    vote.sync.any.pred {tmp_pred2}, {tmp_pred}, {mask};')
                    self._lines.append(f'    selp.s32 {dest}, 1, 0, {tmp_pred2};')
            elif inst.func == '__syncthreads':
                self._lines.append('    bar.sync 0;')
            elif inst.func in ('threadIdx.x', 'threadIdx.y', 'threadIdx.z'):
                sr_map = {'threadIdx.x': 'tid.x', 'threadIdx.y': 'tid.y',
                          'threadIdx.z': 'tid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            elif inst.func in ('blockIdx.x', 'blockIdx.y', 'blockIdx.z'):
                sr_map = {'blockIdx.x': 'ctaid.x', 'blockIdx.y': 'ctaid.y',
                          'blockIdx.z': 'ctaid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            elif inst.func in ('blockDim.x', 'blockDim.y', 'blockDim.z'):
                sr_map = {'blockDim.x': 'ntid.x', 'blockDim.y': 'ntid.y',
                          'blockDim.z': 'ntid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            elif inst.func in ('gridDim.x', 'gridDim.y', 'gridDim.z'):
                sr_map = {'gridDim.x': 'nctaid.x', 'gridDim.y': 'nctaid.y',
                          'gridDim.z': 'nctaid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            else:
                # Math intrinsics and other function calls
                _f32_approx_unary = {
                    'sqrtf': 'sqrt.approx.f32', 'sqrt': 'sqrt.approx.f32',
                    'rsqrtf': 'rsqrt.approx.f32', 'rsqrt': 'rsqrt.approx.f32',
                    'rcpf': 'rcp.approx.f32',
                    'sinf': 'sin.approx.f32', 'sin': 'sin.approx.f32',
                    'cosf': 'cos.approx.f32', 'cos': 'cos.approx.f32',
                    # exp2/log2 map directly to PTX ex2/lg2 (base-2, no scaling needed)
                    'exp2f': 'ex2.approx.f32', 'exp2': 'ex2.approx.f32',
                    'log2f': 'lg2.approx.f32', 'log2': 'lg2.approx.f32',
                    # Rounding-mode sqrt/rcp variants
                    '__fsqrt_rn': 'sqrt.rn.f32', '__fsqrt_rd': 'sqrt.rm.f32',
                    '__fsqrt_ru': 'sqrt.rp.f32', '__fsqrt_rz': 'sqrt.rz.f32',
                    '__frcp_rn':  'rcp.rn.f32',  '__frcp_rd':  'rcp.rm.f32',
                    '__frcp_ru':  'rcp.rp.f32',  '__frcp_rz':  'rcp.rz.f32',
                    '__frsqrt_rn': 'rsqrt.approx.f32',
                }
                # f64 variants: only sqrt.rn.f64 and rsqrt.approx.f64 are direct PTX.
                # All other double-precision transcendentals require cvt→f32→approx→cvt.
                _f64_direct_unary = {
                    'sqrt':  'sqrt.rn.f64',
                    'rsqrt': 'rsqrt.approx.f64',
                    'fabs':  'abs.f64',
                    'floor': 'cvt.rmi.f64.f64', 'ceil': 'cvt.rpi.f64.f64',
                    'trunc': 'cvt.rzi.f64.f64', 'round': 'cvt.rni.f64.f64',
                }
                # f64 fallback: these don't have f64 PTX instructions on SM_120.
                # Emit: cvt.rn.f32.f64 → approx.f32 → cvt.f64.f32 (imprecise but valid).
                _f64_via_f32_approx = {
                    'sin': 'sin.approx.f32', 'cos': 'cos.approx.f32',
                    'exp2': 'ex2.approx.f32', 'log2': 'lg2.approx.f32',
                }
                # Two-step f32 scale then ex2/lg2 (f64 input cast to f32 first).
                _f64_via_f32_scaled = {
                    'exp':   ('ex2.approx.f32', '0f3FB8AA3B'),
                    'log':   ('lg2.approx.f32', '0f3F317218'),
                    'log10': ('lg2.approx.f32', '0f3E9A209B'),
                    'exp10': ('ex2.approx.f32', '0f40549A78'),
                }
                # (unused placeholders — kept for clarity)
                _f64_scaled_unary = {}
                _f64_approx_unary = {}
                # expf(x) = 2^(x*log2e);  logf(x) = lg2(x)*ln2
                # log10f(x) = lg2(x)*log10_2;  exp10f(x) = 2^(x*log2_10)
                # These require two PTX instructions.
                _f32_scaled_unary = {
                    'expf':   ('ex2.approx.f32', '0f3FB8AA3B'),  # * log2(e)
                    'exp':    ('ex2.approx.f32', '0f3FB8AA3B'),
                    'logf':   ('lg2.approx.f32', '0f3F317218'),  # * ln(2)
                    'log':    ('lg2.approx.f32', '0f3F317218'),
                    'log10f': ('lg2.approx.f32', '0f3E9A209B'),  # * log10(2)
                    'log10':  ('lg2.approx.f32', '0f3E9A209B'),
                    'exp10f': ('ex2.approx.f32', '0f40549A78'),  # * log2(10)
                    'exp10':  ('ex2.approx.f32', '0f40549A78'),
                }
                _f32_exact_unary = {
                    'fabsf': 'abs.f32', 'fabs': 'abs.f32',
                    'floorf': 'cvt.rmi.f32.f32', 'floor': 'cvt.rmi.f32.f32',
                    'ceilf': 'cvt.rpi.f32.f32', 'ceil': 'cvt.rpi.f32.f32',
                    'truncf': 'cvt.rzi.f32.f32', 'trunc': 'cvt.rzi.f32.f32',
                    'roundf': 'cvt.rni.f32.f32', 'round': 'cvt.rni.f32.f32',
                }
                _f32_binary = {
                    'fminf': 'min.f32', 'fmin': 'min.f32',
                    'fmaxf': 'max.f32', 'fmax': 'max.f32',
                }
                # Detect if first argument is f64 — redirect to f64 intrinsics.
                _first_arg_is_f64 = (
                    inst.args
                    and isinstance(inst.args[0], Value)
                    and isinstance(inst.args[0].ty, ScalarTy)
                    and inst.args[0].ty.scalar == ScalarType.DOUBLE
                )
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                if inst.func in ('__nanosleep',):
                    # Turing+ warp-level sleep: nanosleep.approx.u32 delay_ns
                    ns_arg = inst.args[0] if inst.args else None
                    ns = self._operand(ns_arg) if ns_arg is not None else '0'
                    self._lines.append(f'    nanosleep.u32 {ns};')
                elif inst.func in ('__trap',):
                    self._lines.append('    trap;')
                elif inst.func in ('__brkpt',):
                    self._lines.append('    brkpt;')
                elif inst.func in ('clock64', '__clock64'):
                    # 64-bit hardware cycle counter
                    self._lines.append(f'    mov.u64 {dest}, %clock64;')
                elif inst.func in ('clock',):
                    # 32-bit hardware cycle counter
                    self._lines.append(f'    mov.u32 {dest}, %clock;')
                elif inst.func in ('__threadfence',):
                    # Global memory fence (visible to all threads)
                    self._lines.append('    membar.gl;')
                elif inst.func in ('__threadfence_block',):
                    # Block-level memory fence
                    self._lines.append('    membar.cta;')
                elif inst.func in ('__threadfence_system',):
                    # System-wide memory fence (GPU + CPU)
                    self._lines.append('    membar.sys;')
                elif inst.func in ('__syncwarp',):
                    # Warp-level synchronization
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    self._lines.append(f'    bar.warp.sync {mask};')
                elif inst.func in ('__activemask',):
                    # Returns bitmask of active lanes in warp
                    self._lines.append(f'    activemask.b32 {dest};')
                elif inst.func in ('__syncthreads_count', '__syncthreads_and',
                                   '__syncthreads_or'):
                    # Barrier + integer reduction across a CTA.
                    # PTX: bar.red.op.type d, a, b  (a=barrier name, b=predicate)
                    src_arg = inst.args[0] if inst.args else None
                    src = self._operand(src_arg) if src_arg is not None else '0'
                    n = inst.dest.id if inst.dest else 0
                    # If source is already a predicate register, use it directly;
                    # otherwise convert integer to predicate via setp.ne.
                    src_is_pred = (isinstance(src_arg, Value) and src_arg.id in self._pred_ids)
                    if src_is_pred:
                        p_tmp = src_arg  # reuse existing pred register
                    else:
                        p_tmp = kernel.new_value(f'_sp_{n}', ScalarTy(ScalarType.BOOL))
                        self._pred_ids.add(p_tmp.id)
                        self._lines.append(f'    setp.ne.s32 {self._reg(p_tmp)}, {src}, 0;')
                    if inst.func == '__syncthreads_count':
                        self._lines.append(f'    bar.red.popc.u32 {dest}, 0, {self._reg(p_tmp)};')
                    elif inst.func == '__syncthreads_and':
                        # bar.red.and/or.pred result is a predicate; convert to int
                        p_out = kernel.new_value(f'_sp_out_{n}', ScalarTy(ScalarType.BOOL))
                        self._pred_ids.add(p_out.id)
                        self._lines.append(f'    bar.red.and.pred {self._reg(p_out)}, 0, {self._reg(p_tmp)};')
                        self._lines.append(f'    selp.s32 {dest}, 1, 0, {self._reg(p_out)};')
                    else:
                        p_out = kernel.new_value(f'_sp_out_{n}', ScalarTy(ScalarType.BOOL))
                        self._pred_ids.add(p_out.id)
                        self._lines.append(f'    bar.red.or.pred {self._reg(p_out)}, 0, {self._reg(p_tmp)};')
                        self._lines.append(f'    selp.s32 {dest}, 1, 0, {self._reg(p_out)};')
                elif inst.func in ('__popc', '__popcll'):
                    # Population count: PTX type = source width (dest is always 32-bit).
                    # popc.b32 dest, src32  or  popc.b64 dest, src64
                    src_arg = inst.args[0] if inst.args else None
                    src_ty = src_arg.ty if isinstance(src_arg, (Value, Const)) else INT32
                    ptx_ty = 'b64' if (isinstance(src_ty, ScalarTy) and src_ty.size == 8) else 'b32'
                    src = self._operand(src_arg) if src_arg is not None else '0'
                    self._lines.append(f'    popc.{ptx_ty} {dest}, {src};')
                elif inst.func in ('__clz', '__clzll'):
                    # Count leading zeros: PTX type = source width (dest is always 32-bit).
                    src_arg = inst.args[0] if inst.args else None
                    src_ty = src_arg.ty if isinstance(src_arg, (Value, Const)) else INT32
                    ptx_ty = 'b64' if (isinstance(src_ty, ScalarTy) and src_ty.size == 8) else 'b32'
                    src = self._operand(src_arg) if src_arg is not None else '0'
                    self._lines.append(f'    clz.{ptx_ty} {dest}, {src};')
                elif inst.func in ('__brev', '__brevll'):
                    # Bit reversal: PTX type = source width = dest width.
                    # brev.b32 for __brev, brev.b64 for __brevll.
                    src_arg = inst.args[0] if inst.args else None
                    src_ty = src_arg.ty if isinstance(src_arg, (Value, Const)) else INT32
                    ptx_ty = 'b64' if (isinstance(src_ty, ScalarTy) and src_ty.size == 8) else 'b32'
                    src = self._operand(src_arg) if src_arg is not None else '0'
                    self._lines.append(f'    brev.{ptx_ty} {dest}, {src};')
                elif inst.func in ('__ffs', '__ffsll'):
                    # __ffs(x): 1-indexed position of least significant 1-bit (0 if x==0)
                    # PTX: neg → and (isolate LSB) → bfind.u32 → add 1
                    # Zero case: bfind(0)=0xFFFFFFFF, +1 overflows to 0 — correct!
                    src_arg = inst.args[0] if inst.args else None
                    src_ty = src_arg.ty if isinstance(src_arg, (Value, Const)) else INT32
                    is64 = isinstance(src_ty, ScalarTy) and src_ty.size == 8
                    ptx_ty = 'b64' if is64 else 'b32'
                    neg_ty = 's64' if is64 else 's32'
                    src = self._operand(src_arg) if src_arg is not None else '0'
                    n = inst.dest.id if inst.dest else 0
                    tmp_neg = kernel.new_value(f'_ffs_neg_{n}', INT32)
                    tmp_lsb = kernel.new_value(f'_ffs_lsb_{n}', INT32)
                    tmp_pos = kernel.new_value(f'_ffs_pos_{n}', UINT32)
                    self._lines.append(f'    neg.{neg_ty} {self._reg(tmp_neg)}, {src};')
                    self._lines.append(f'    and.{ptx_ty} {self._reg(tmp_lsb)}, {src}, {self._reg(tmp_neg)};')
                    self._lines.append(f'    bfind.u32 {self._reg(tmp_pos)}, {self._reg(tmp_lsb)};')
                    self._lines.append(f'    add.s32 {dest}, {self._reg(tmp_pos)}, 1;')
                elif _first_arg_is_f64 and inst.func in _f64_direct_unary:
                    # Direct f64 PTX instruction (sqrt.rn, rsqrt.approx, abs, cvt rounding).
                    ptx_op = _f64_direct_unary[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    {ptx_op} {dest}, {src};')
                elif _first_arg_is_f64 and inst.func in _f64_via_f32_approx:
                    # No f64 PTX approx: downcast to f32, compute, upcast back to f64.
                    ptx_op = _f64_via_f32_approx[inst.func]
                    n = inst.dest.id if inst.dest else 0
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    tmp32 = kernel.new_value(f'_d2f_{n}', FLOAT)
                    res32 = kernel.new_value(f'_f32r_{n}', FLOAT)
                    self._lines.append(f'    cvt.rn.f32.f64 {self._reg(tmp32)}, {src};')
                    self._lines.append(f'    {ptx_op} {self._reg(res32)}, {self._reg(tmp32)};')
                    self._lines.append(f'    cvt.f64.f32 {dest}, {self._reg(res32)};')
                elif _first_arg_is_f64 and inst.func in _f64_via_f32_scaled:
                    # Two-step via f32: downcast, scale, ex2/lg2 approx, upcast.
                    ptx_op, scale_hex = _f64_via_f32_scaled[inst.func]
                    n = inst.dest.id if inst.dest else 0
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    tmp32 = kernel.new_value(f'_d2f_{n}', FLOAT)
                    scaled = kernel.new_value(f'_scl_{n}', FLOAT)
                    res32  = kernel.new_value(f'_f32r_{n}', FLOAT)
                    self._lines.append(f'    cvt.rn.f32.f64 {self._reg(tmp32)}, {src};')
                    self._lines.append(f'    mul.f32 {self._reg(scaled)}, {self._reg(tmp32)}, {scale_hex};')
                    self._lines.append(f'    {ptx_op} {self._reg(res32)}, {self._reg(scaled)};')
                    self._lines.append(f'    cvt.f64.f32 {dest}, {self._reg(res32)};')
                elif inst.func in _f32_scaled_unary:
                    # Two-instruction form: scale input then apply base-2 op.
                    # e.g. expf(x): tmp = x * log2e; dest = ex2.approx(tmp)
                    ptx_op, scale_hex = _f32_scaled_unary[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    tmp = kernel.new_value(f'_scale_{inst.dest.id if inst.dest else 0}',
                                          FLOAT)
                    self._lines.append(
                        f'    mul.f32 {self._reg(tmp)}, {src}, {scale_hex};')
                    self._lines.append(
                        f'    {ptx_op} {dest}, {self._reg(tmp)};')
                elif inst.func in _f32_approx_unary or inst.func in _f32_exact_unary:
                    ptx_op = (_f32_approx_unary.get(inst.func)
                              or _f32_exact_unary.get(inst.func))
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    {ptx_op} {dest}, {src};')
                elif inst.func in _f32_binary:
                    if _first_arg_is_f64:
                        ptx_op = _f32_binary[inst.func].replace('.f32', '.f64')
                    else:
                        ptx_op = _f32_binary[inst.func]
                    a = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    self._lines.append(f'    {ptx_op} {dest}, {a}, {b};')
                elif inst.func in ('fmaf', 'fma',
                                   '__fmaf_rn', '__fmaf_rd', '__fmaf_ru', '__fmaf_rz',
                                   '__fma_rn', '__fma_rd', '__fma_ru', '__fma_rz'):
                    # Fused multiply-add: fma(a, b, c) = a*b + c (single rounding)
                    fma_ty = 'f64' if _first_arg_is_f64 else 'f32'
                    _fn = inst.func
                    _rnd = _fn[-2:] if _fn.endswith(('rn','rd','ru','rz')) else 'rn'
                    a = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    c = self._operand(inst.args[2]) if len(inst.args) > 2 else '0f00000000'
                    self._lines.append(f'    fma.{_rnd}.{fma_ty} {dest}, {a}, {b}, {c};')
                elif inst.func in ('fmodf', 'fmod'):
                    # fmod(x, y) = x - trunc(x/y)*y — no direct PTX opcode
                    # Uses: div.approx, cvt.rzi (truncate-toward-zero), mul, sub
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    y = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    q = kernel.new_value(f'_fmod_q_{n}', FLOAT)
                    qt = kernel.new_value(f'_fmod_qt_{n}', FLOAT)
                    qy = kernel.new_value(f'_fmod_qy_{n}', FLOAT)
                    self._lines.append(f'    div.approx.f32 {self._reg(q)}, {x}, {y};')
                    self._lines.append(f'    cvt.rzi.f32.f32 {self._reg(qt)}, {self._reg(q)};')
                    self._lines.append(f'    mul.f32 {self._reg(qy)}, {self._reg(qt)}, {y};')
                    self._lines.append(f'    sub.f32 {dest}, {x}, {self._reg(qy)};')
                elif inst.func in ('tanf', 'tan'):
                    # tan(x) = sin(x)/cos(x) — no direct PTX tan opcode
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    s = kernel.new_value(f'_tan_s_{n}', FLOAT)
                    c = kernel.new_value(f'_tan_c_{n}', FLOAT)
                    self._lines.append(f'    sin.approx.f32 {self._reg(s)}, {x};')
                    self._lines.append(f'    cos.approx.f32 {self._reg(c)}, {x};')
                    self._lines.append(f'    div.approx.f32 {dest}, {self._reg(s)}, {self._reg(c)};')
                elif inst.func in ('abs',):
                    ty = inst.dest.ty if inst.dest else INT32
                    ptx_ty = _ptx_type(ty)
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    abs.{ptx_ty} {dest}, {src};')
                elif inst.func in ('min', 'max'):
                    ty = inst.dest.ty if inst.dest else INT32
                    ptx_ty = _ptx_type(ty)
                    ptx_op = 'min' if inst.func == 'min' else 'max'
                    a = self._operand(inst.args[0], ptx_ty) if inst.args else '0'
                    b = self._operand(inst.args[1], ptx_ty) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    {ptx_op}.{ptx_ty} {dest}, {a}, {b};')
                elif inst.func in ('isnan', 'isinf', 'isfinite',
                                   '__isnan', '__isinf', '__isfinite',
                                   '__isnanf', '__isinff', '__isfinitef'):
                    # testp.{property}.f32 pred, src; selp.s32 dest, 1, 0, pred
                    _testp_map = {
                        'isnan': 'notanumber', '__isnan': 'notanumber', '__isnanf': 'notanumber',
                        'isinf': 'infinite',   '__isinf': 'infinite',   '__isinff': 'infinite',
                        'isfinite': 'finite',  '__isfinite': 'finite',  '__isfinitef': 'finite',
                    }
                    prop = _testp_map[inst.func]
                    src_arg = inst.args[0] if inst.args else None
                    src_ty = src_arg.ty if isinstance(src_arg, (Value, Const)) else FLOAT
                    ptx_float_ty = 'f64' if (isinstance(src_ty, ScalarTy) and src_ty.size == 8) else 'f32'
                    src = self._operand(src_arg) if src_arg is not None else '0f00000000'
                    p_tmp = kernel.new_value(f'_tp_{inst.dest.id if inst.dest else 0}',
                                            ScalarTy(ScalarType.BOOL))
                    self._pred_ids.add(p_tmp.id)
                    self._lines.append(f'    testp.{prop}.{ptx_float_ty} {self._reg(p_tmp)}, {src};')
                    self._lines.append(f'    selp.s32 {dest}, 1, 0, {self._reg(p_tmp)};')
                elif inst.func in ('__float2int_rn', '__float2int_rd',
                                   '__float2int_ru', '__float2int_rz'):
                    _rnd_map = {'__float2int_rn': 'rni', '__float2int_rd': 'rmi',
                                '__float2int_ru': 'rpi', '__float2int_rz': 'rzi'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.{rnd}.s32.f32 {dest}, {src};')
                elif inst.func in ('__float2uint_rn', '__float2uint_rd',
                                   '__float2uint_ru', '__float2uint_rz'):
                    _rnd_map = {'__float2uint_rn': 'rni', '__float2uint_rd': 'rmi',
                                '__float2uint_ru': 'rpi', '__float2uint_rz': 'rzi'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.{rnd}.u32.f32 {dest}, {src};')
                elif inst.func in ('__float2ll_rn', '__float2ll_rd',
                                   '__float2ll_ru', '__float2ll_rz'):
                    _rnd_map = {'__float2ll_rn': 'rni', '__float2ll_rd': 'rmi',
                                '__float2ll_ru': 'rpi', '__float2ll_rz': 'rzi'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.{rnd}.s64.f32 {dest}, {src};')
                elif inst.func in ('__float2ull_rn', '__float2ull_rd',
                                   '__float2ull_ru', '__float2ull_rz'):
                    _rnd_map = {'__float2ull_rn': 'rni', '__float2ull_rd': 'rmi',
                                '__float2ull_ru': 'rpi', '__float2ull_rz': 'rzi'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.{rnd}.u64.f32 {dest}, {src};')
                elif inst.func in ('__double2int_rn', '__double2int_rz'):
                    rnd = 'rni' if inst.func == '__double2int_rn' else 'rzi'
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    cvt.{rnd}.s32.f64 {dest}, {src};')
                elif inst.func in ('__double2uint_rn', '__double2uint_rz'):
                    rnd = 'rni' if inst.func.endswith('rn') else 'rzi'
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    cvt.{rnd}.u32.f64 {dest}, {src};')
                elif inst.func in ('__double2ll_rn', '__double2ll_rz'):
                    rnd = 'rni' if inst.func.endswith('rn') else 'rzi'
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    cvt.{rnd}.s64.f64 {dest}, {src};')
                elif inst.func in ('__double2ull_rn', '__double2ull_rz'):
                    rnd = 'rni' if inst.func.endswith('rn') else 'rzi'
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    cvt.{rnd}.u64.f64 {dest}, {src};')
                elif inst.func in ('__int2float_rn', '__int2float_rd',
                                   '__int2float_ru', '__int2float_rz'):
                    _rnd_map = {'__int2float_rn': 'rn', '__int2float_rd': 'rm',
                                '__int2float_ru': 'rp', '__int2float_rz': 'rz'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    cvt.{rnd}.f32.s32 {dest}, {src};')
                elif inst.func in ('__uint2float_rn', '__uint2float_rd',
                                   '__uint2float_ru', '__uint2float_rz'):
                    _rnd_map = {'__uint2float_rn': 'rn', '__uint2float_rd': 'rm',
                                '__uint2float_ru': 'rp', '__uint2float_rz': 'rz'}
                    rnd = _rnd_map[inst.func]
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    cvt.{rnd}.f32.u32 {dest}, {src};')
                elif inst.func in ('__ll2float_rn', '__ll2float_rz'):
                    rnd = 'rn' if inst.func == '__ll2float_rn' else 'rz'
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    cvt.{rnd}.f32.s64 {dest}, {src};')
                elif inst.func in ('__ull2float_rn', '__ull2float_rz'):
                    rnd = 'rn' if inst.func.endswith('rn') else 'rz'
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    cvt.{rnd}.f32.u64 {dest}, {src};')
                elif inst.func in ('__int2double_rn', '__ll2double_rn',
                                   '__uint2double_rn', '__ull2double_rn'):
                    if 'ull' in inst.func: src_ty_str = 'u64'
                    elif 'll' in inst.func: src_ty_str = 's64'
                    elif 'uint' in inst.func: src_ty_str = 'u32'
                    else: src_ty_str = 's32'
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    cvt.rn.f64.{src_ty_str} {dest}, {src};')
                elif inst.func in ('__float_as_int', '__float_as_uint'):
                    # Bit-reinterpret float → int32: mov.b32 %r, %f
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    mov.b32 {dest}, {src};')
                elif inst.func in ('__int_as_float', '__uint_as_float'):
                    # Bit-reinterpret int32 → float: mov.b32 %f, %r
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    mov.b32 {dest}, {src};')
                elif inst.func in ('__double_as_longlong', '__double_as_ulonglong'):
                    # Bit-reinterpret f64 → i64: mov.b64 %rd, %fd
                    src = self._operand(inst.args[0]) if inst.args else '0d0000000000000000'
                    self._lines.append(f'    mov.b64 {dest}, {src};')
                elif inst.func in ('__longlong_as_double', '__ulonglong_as_double'):
                    # Bit-reinterpret i64 → f64: mov.b64 %fd, %rd
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    self._lines.append(f'    mov.b64 {dest}, {src};')
                elif inst.func in ('__sad', '__usad'):
                    # Sum of absolute differences: sad.{s32|u32} dest, a, b, accum
                    ptx_ty = 'u32' if inst.func == '__usad' else 's32'
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    acc = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                    self._lines.append(f'    sad.{ptx_ty} {dest}, {a}, {b}, {acc};')
                elif inst.func == 'warpSize':
                    # CUDA warp size is always 32 (SM constant).
                    self._lines.append(f'    mov.u32 {dest}, 32;')
                elif inst.func in ('__dp4a', '__dp4a_u', '__dp4a_su', '__dp4a_us'):
                    # Byte dot-product accumulate: dp4a.TYPE.TYPE dest, a, b, c
                    # __dp4a → s32.s32; __dp4a_u → u32.u32; _su → s32.u32; _us → u32.s32
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    c = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                    if inst.func == '__dp4a_u':
                        self._lines.append(f'    dp4a.u32.u32 {dest}, {a}, {b}, {c};')
                    elif inst.func == '__dp4a_su':
                        self._lines.append(f'    dp4a.s32.u32 {dest}, {a}, {b}, {c};')
                    elif inst.func == '__dp4a_us':
                        self._lines.append(f'    dp4a.u32.s32 {dest}, {a}, {b}, {c};')
                    else:
                        self._lines.append(f'    dp4a.s32.s32 {dest}, {a}, {b}, {c};')
                elif inst.func in ('__reduce_add_sync',):
                    # redux.sync.add.s32 dest, val, mask
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.add.s32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_min_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.min.s32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_max_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.max.s32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_and_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.and.b32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_or_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.or.b32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_xor_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.xor.b32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_umin_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.min.u32 {dest}, {val}, {mask};')
                elif inst.func in ('__reduce_umax_sync',):
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    redux.sync.max.u32 {dest}, {val}, {mask};')
                elif inst.func in ('__match_any_sync',):
                    # match.any.sync.b32 dest, val, mask
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    match.any.sync.b32 {dest}, {val}, {mask};')
                elif inst.func in ('__match_all_sync',):
                    # match.all.sync.b32 dest, val, mask
                    # 3rd arg (&pred) is an output pointer — not representable in our IR;
                    # omit the predicate output and just return the match mask.
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    val  = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(
                        f'    match.all.sync.b32 {dest}, {val}, {mask};')
                elif inst.func in ('__mul24',):
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    mul24.lo.s32 {dest}, {a}, {b};')
                elif inst.func in ('__umul24',):
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    mul24.lo.u32 {dest}, {a}, {b};')
                elif inst.func in ('__mulhi',):
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    mul.hi.s32 {dest}, {a}, {b};')
                elif inst.func in ('__umulhi',):
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    self._lines.append(f'    mul.hi.u32 {dest}, {a}, {b};')
                elif inst.func in ('__hadd', '__hadd_rn', '__hadd_sat'):
                    # Dispatch: __hadd with HALF result → fp16 add; otherwise integer halving.
                    dest_ty = inst.dest.ty if inst.dest else INT32
                    if isinstance(dest_ty, ScalarTy) and dest_ty.scalar == ScalarType.HALF:
                        a = self._operand(inst.args[0]) if inst.args else '0h0000'
                        b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                        sat = '.sat' if inst.func.endswith('_sat') else ''
                        self._lines.append(f'    add.rn{sat}.f16 {dest}, {a}, {b};')
                    else:
                        # Halving add (no overflow): (a + b) >> 1 (signed arithmetic shift)
                        a = self._operand(inst.args[0]) if inst.args else '0'
                        b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                        n = inst.dest.id if inst.dest else 0
                        tmp = kernel.new_value(f'_hadd_tmp_{n}', INT32)
                        self._lines.append(f'    add.s32 {self._reg(tmp)}, {a}, {b};')
                        self._lines.append(f'    shr.s32 {dest}, {self._reg(tmp)}, 1;')
                elif inst.func in ('__rhadd',):
                    # Rounding halving add: (a + b + 1) >> 1 (unsigned)
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    n = inst.dest.id if inst.dest else 0
                    tmp = kernel.new_value(f'_rhadd_tmp_{n}', UINT32)
                    self._lines.append(f'    add.u32 {self._reg(tmp)}, {a}, {b};')
                    self._lines.append(f'    add.u32 {self._reg(tmp)}, {self._reg(tmp)}, 1;')
                    self._lines.append(f'    shr.u32 {dest}, {self._reg(tmp)}, 1;')
                elif inst.func in ('__byte_perm',):
                    # Byte permutation: prmt.b32 dest, a, b, selector
                    a = self._operand(inst.args[0]) if inst.args else '0'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    sel = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                    self._lines.append(f'    prmt.b32 {dest}, {a}, {b}, {sel};')
                elif inst.func in ('__funnelshift_l', '__funnelshift_lc'):
                    # Funnel shift left: shf.l.wrap.b32 dest, lo, hi, shift
                    lo = self._operand(inst.args[0]) if inst.args else '0'
                    hi = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    sh = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                    clamp = 'clamp' if inst.func.endswith('c') else 'wrap'
                    self._lines.append(f'    shf.l.{clamp}.b32 {dest}, {lo}, {hi}, {sh};')
                elif inst.func in ('__funnelshift_r', '__funnelshift_rc'):
                    # Funnel shift right: shf.r.wrap.b32 dest, lo, hi, shift
                    lo = self._operand(inst.args[0]) if inst.args else '0'
                    hi = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                    sh = self._operand(inst.args[2]) if len(inst.args) > 2 else '0'
                    clamp = 'clamp' if inst.func.endswith('c') else 'wrap'
                    self._lines.append(f'    shf.r.{clamp}.b32 {dest}, {lo}, {hi}, {sh};')
                elif inst.func in ('__ushort_as_half', '__short_as_half'):
                    # Bit-reinterpret u16/s16 → f16.
                    # Source is typically a b32 register (ld.global.u16 zero-extends to
                    # b32).  PTX mov.b16 requires same-width operands, so use vector
                    # unpack: mov.b32 {h_dest, h_discard}, r_src — lower 16 bits → dest.
                    if inst.args:
                        src = self._operand(inst.args[0])
                        if src.startswith('%r') or src.startswith('%rd'):
                            discard = kernel.new_value(
                                f'_half_discard_{inst.dest.id}', HALF)
                            self._lines.append(
                                f'    mov.b32 {{{dest}, {self._reg(discard)}}}, {src};')
                        else:
                            self._lines.append(f'    mov.b16 {dest}, {src};')
                    else:
                        self._lines.append(f'    mov.b16 {dest}, 0;')
                elif inst.func in ('__half_as_ushort', '__half_as_short'):
                    # Bit-reinterpret f16 → u32 (zero-extended).
                    # Destination is b32 register; PTX mov.b16 requires same-width
                    # operands.  Use vector pack: mov.b32 r_dest, {h_src, h_zero}.
                    if inst.args:
                        src = self._operand(inst.args[0])
                        if src.startswith('%h'):
                            zero_h = kernel.new_value(
                                f'_half_zero_{inst.dest.id}', HALF)
                            self._lines.append(
                                f'    mov.b16 {self._reg(zero_h)}, 0;')
                            self._lines.append(
                                f'    mov.b32 {dest}, {{{src}, {self._reg(zero_h)}}};')
                        else:
                            self._lines.append(f'    mov.b16 {dest}, {src};')
                    else:
                        self._lines.append(f'    mov.b32 {dest}, 0;')
                elif inst.func in ('__int2half_rn', '__uint2half_rn',
                                    '__short2half_rn', '__ushort2half_rn',
                                    '__ll2half_rn', '__ull2half_rn'):
                    # Convert integer → f16
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    if isinstance(inst.args[0] if inst.args else None, Value):
                        src_ptx = _ptx_type(inst.args[0].ty)
                    else:
                        src_ptx = 's32'
                    self._lines.append(f'    cvt.rn.f16.{src_ptx} {dest}, {src};')
                elif inst.func in ('__int2half_rz', '__uint2half_rz',
                                    '__ll2half_rz', '__ull2half_rz'):
                    src = self._operand(inst.args[0]) if inst.args else '0'
                    src_ptx = _ptx_type(inst.args[0].ty) if inst.args and isinstance(inst.args[0], Value) else 's32'
                    self._lines.append(f'    cvt.rz.f16.{src_ptx} {dest}, {src};')
                elif inst.func in ('__half2int_rn', '__half2uint_rn',
                                    '__half2short_rn', '__half2ushort_rn',
                                    '__half2ll_rn', '__half2ull_rn'):
                    # Convert f16 → integer (round nearest integer)
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    dst_ptx = _ptx_type(inst.dest.ty) if inst.dest else 's32'
                    self._lines.append(f'    cvt.rni.{dst_ptx}.f16 {dest}, {src};')
                elif inst.func in ('__half2int_rz', '__half2uint_rz',
                                    '__half2short_rz', '__half2ushort_rz',
                                    '__half2ll_rz', '__half2ull_rz'):
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    dst_ptx = _ptx_type(inst.dest.ty) if inst.dest else 's32'
                    self._lines.append(f'    cvt.rzi.{dst_ptx}.f16 {dest}, {src};')
                elif inst.func in ('__float2half', '__float2half_rn'):
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.rn.f16.f32 {dest}, {src};')
                elif inst.func in ('__float2half_rd',):
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.rm.f16.f32 {dest}, {src};')
                elif inst.func in ('__float2half_ru',):
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.rp.f16.f32 {dest}, {src};')
                elif inst.func in ('__float2half_rz',):
                    src = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    self._lines.append(f'    cvt.rz.f16.f32 {dest}, {src};')
                elif inst.func in ('__half2float', '__low2float', '__high2float'):
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    self._lines.append(f'    cvt.f32.f16 {dest}, {src};')
                elif inst.func in ('__hmul', '__hmul_rn'):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    self._lines.append(f'    mul.rn.f16 {dest}, {a}, {b};')
                elif inst.func in ('__hmul_sat',):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    self._lines.append(f'    mul.rn.sat.f16 {dest}, {a}, {b};')
                elif inst.func in ('__hsub', '__hsub_rn'):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    self._lines.append(f'    sub.rn.f16 {dest}, {a}, {b};')
                elif inst.func in ('__hdiv',):
                    # PTX has no div.f16 or rcp.f16; promote b to f32, compute rcp,
                    # demote back to f16, then mul.f16.
                    n = inst.dest.id if inst.dest else 0
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h3C00'
                    f32_b   = kernel.new_value(f'_hdiv_b_{n}', FLOAT)
                    f32_rcp = kernel.new_value(f'_hdiv_r_{n}', FLOAT)
                    h_rcp   = kernel.new_value(f'_hdiv_h_{n}', HALF)
                    self._lines.append(f'    cvt.f32.f16 {self._reg(f32_b)}, {b};')
                    self._lines.append(f'    rcp.approx.f32 {self._reg(f32_rcp)}, {self._reg(f32_b)};')
                    self._lines.append(f'    cvt.rn.f16.f32 {self._reg(h_rcp)}, {self._reg(f32_rcp)};')
                    self._lines.append(f'    mul.rn.f16 {dest}, {a}, {self._reg(h_rcp)};')
                elif inst.func in ('__hfmin', '__hmin'):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    self._lines.append(f'    min.f16 {dest}, {a}, {b};')
                elif inst.func in ('__hfmax', '__hmax'):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    self._lines.append(f'    max.f16 {dest}, {a}, {b};')
                elif inst.func in ('__hfma', '__hfma_sat'):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    c = self._operand(inst.args[2]) if len(inst.args) > 2 else '0h0000'
                    sat = '.sat' if inst.func.endswith('_sat') else ''
                    self._lines.append(f'    fma.rn{sat}.f16 {dest}, {a}, {b}, {c};')
                elif inst.func in ('__hfma_relu',):
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    c = self._operand(inst.args[2]) if len(inst.args) > 2 else '0h0000'
                    self._lines.append(f'    fma.rn.relu.f16 {dest}, {a}, {b}, {c};')
                elif inst.func in ('__habs',):
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    self._lines.append(f'    abs.f16 {dest}, {src};')
                elif inst.func in ('__hneg',):
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    self._lines.append(f'    neg.f16 {dest}, {src};')
                elif inst.func in ('__hrcp', '__hsqrt', '__hrsqrt', '__hexp', '__hlog',
                                    '__hceil', '__hfloor', '__hrint', '__htrunc',
                                    '__hcos', '__hsin', '__hlog2', '__hlog10'):
                    # PTX has no native f16 versions; promote to f32, apply, demote.
                    n = inst.dest.id if inst.dest else 0
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    f32_in  = kernel.new_value(f'_hf32in_{n}', FLOAT)
                    f32_out = kernel.new_value(f'_hf32out_{n}', FLOAT)
                    self._lines.append(f'    cvt.f32.f16 {self._reg(f32_in)}, {src};')
                    if inst.func == '__hrcp':
                        self._lines.append(f'    rcp.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func == '__hsqrt':
                        self._lines.append(f'    sqrt.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func == '__hrsqrt':
                        self._lines.append(f'    rsqrt.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func == '__hexp':
                        # e^x = 2^(x*log2e)
                        scale = kernel.new_value(f'_hscale_{n}', FLOAT)
                        self._lines.append(f'    mul.f32 {self._reg(scale)}, {self._reg(f32_in)}, 0f3FB8AA3B;')
                        self._lines.append(f'    ex2.approx.f32 {self._reg(f32_out)}, {self._reg(scale)};')
                    elif inst.func == '__hlog':
                        # ln(x) = lg2(x) * ln(2)
                        lg2 = kernel.new_value(f'_hlg2_{n}', FLOAT)
                        self._lines.append(f'    lg2.approx.f32 {self._reg(lg2)}, {self._reg(f32_in)};')
                        self._lines.append(f'    mul.f32 {self._reg(f32_out)}, {self._reg(lg2)}, 0f3F317218;')
                    elif inst.func == '__hlog2':
                        self._lines.append(f'    lg2.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func == '__hlog10':
                        # log10(x) = lg2(x) * log10(2)
                        lg2 = kernel.new_value(f'_hlg2_{n}', FLOAT)
                        self._lines.append(f'    lg2.approx.f32 {self._reg(lg2)}, {self._reg(f32_in)};')
                        self._lines.append(f'    mul.f32 {self._reg(f32_out)}, {self._reg(lg2)}, 0f3E9A209B;')
                    elif inst.func == '__hcos':
                        self._lines.append(f'    cos.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func == '__hsin':
                        self._lines.append(f'    sin.approx.f32 {self._reg(f32_out)}, {self._reg(f32_in)};')
                    elif inst.func in ('__hceil', '__hfloor', '__hrint', '__htrunc'):
                        # Round via cvt to int then back to f32
                        int_tmp = kernel.new_value(f'_hint_{n}', INT32)
                        rmode = {'__hceil': 'rpi', '__hfloor': 'rmi',
                                 '__hrint': 'rni', '__htrunc': 'rzi'}[inst.func]
                        self._lines.append(f'    cvt.{rmode}.s32.f32 {self._reg(int_tmp)}, {self._reg(f32_in)};')
                        self._lines.append(f'    cvt.rn.f32.s32 {self._reg(f32_out)}, {self._reg(int_tmp)};')
                    self._lines.append(f'    cvt.rn.f16.f32 {dest}, {self._reg(f32_out)};')
                elif inst.func in ('__hexp2',):
                    # ex2.approx.f16 is a valid PTX instruction for sm_120
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    self._lines.append(f'    ex2.approx.f16 {dest}, {src};')
                elif inst.func in ('__hgt', '__hlt', '__hge', '__hle', '__heq', '__hne'):
                    _hcmp_map = {
                        '__hgt': 'gt', '__hlt': 'lt', '__hge': 'ge',
                        '__hle': 'le', '__heq': 'eq', '__hne': 'ne',
                    }
                    cmp_op = _hcmp_map[inst.func]
                    a = self._operand(inst.args[0]) if inst.args else '0h0000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0h0000'
                    n = inst.dest.id if inst.dest else 0
                    p_tmp = kernel.new_value(f'_hcmp_{n}', ScalarTy(ScalarType.BOOL))
                    self._pred_ids.add(p_tmp.id)
                    self._lines.append(f'    setp.{cmp_op}.f16 {self._reg(p_tmp)}, {a}, {b};')
                    self._lines.append(f'    selp.s32 {dest}, 1, 0, {self._reg(p_tmp)};')
                elif inst.func in ('__hisnan', '__hisinf'):
                    # PTX has no testp.f16; convert to f32 then test.
                    src = self._operand(inst.args[0]) if inst.args else '0h0000'
                    n = inst.dest.id if inst.dest else 0
                    f32_tmp = kernel.new_value(f'_h2f_{n}', FLOAT)
                    p_tmp   = kernel.new_value(f'_htest_{n}', ScalarTy(ScalarType.BOOL))
                    self._pred_ids.add(p_tmp.id)
                    self._lines.append(f'    cvt.f32.f16 {self._reg(f32_tmp)}, {src};')
                    cond = 'notanumber' if inst.func == '__hisnan' else 'infinite'
                    self._lines.append(f'    testp.{cond}.f32 {self._reg(p_tmp)}, {self._reg(f32_tmp)};')
                    self._lines.append(f'    selp.s32 {dest}, 1, 0, {self._reg(p_tmp)};')
                elif inst.func in ('atanf', 'atan'):
                    # atan(x) — polynomial approx valid for all x via range reduction.
                    # For |x| <= 1: atan(x) ≈ x*(1 - x^2*(1/3 - x^2*(1/5 - x^2/7)))
                    # For |x| > 1:  atan(x) = sign(x)*pi/2 - atan(1/x)
                    # Practical PTX: use a 5-term Maclaurin for |x| <= 1 only (no reduction).
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    x2   = kernel.new_value(f'_atan_x2_{n}', FLOAT)
                    x4   = kernel.new_value(f'_atan_x4_{n}', FLOAT)
                    x6   = kernel.new_value(f'_atan_x6_{n}', FLOAT)
                    p    = kernel.new_value(f'_atan_p_{n}', FLOAT)
                    p2   = kernel.new_value(f'_atan_p2_{n}', FLOAT)
                    self._lines.append(f'    mul.f32 {self._reg(x2)}, {x}, {x};')
                    self._lines.append(f'    mul.f32 {self._reg(x4)}, {self._reg(x2)}, {self._reg(x2)};')
                    self._lines.append(f'    mul.f32 {self._reg(x6)}, {self._reg(x4)}, {self._reg(x2)};')
                    # p = 1 - x2/3 + x4/5 - x6/7  (Maclaurin series coefficients)
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(x6)}, 0fBE124925, 0f3F800000;')  # -1/7
                    self._lines.append(f'    fma.rn.f32 {self._reg(p2)}, {self._reg(x4)}, 0f3E4CCCCD, {self._reg(p)};')  # +1/5
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(x2)}, 0fBEAAAAAB, {self._reg(p2)};')  # -1/3
                    self._lines.append(f'    mul.f32 {dest}, {x}, {self._reg(p)};')
                elif inst.func in ('asinf', 'asin'):
                    # asin(x) = atan(x / sqrt(1 - x*x))
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    x2   = kernel.new_value(f'_asin_x2_{n}', FLOAT)
                    om   = kernel.new_value(f'_asin_om_{n}', FLOAT)
                    sqr  = kernel.new_value(f'_asin_sq_{n}', FLOAT)
                    rat  = kernel.new_value(f'_asin_r_{n}',  FLOAT)
                    # Reuse atan approximation inline
                    ax2  = kernel.new_value(f'_asin_ax2_{n}', FLOAT)
                    ax4  = kernel.new_value(f'_asin_ax4_{n}', FLOAT)
                    ax6  = kernel.new_value(f'_asin_ax6_{n}', FLOAT)
                    p    = kernel.new_value(f'_asin_p_{n}',   FLOAT)
                    p2   = kernel.new_value(f'_asin_p2_{n}',  FLOAT)
                    self._lines.append(f'    mul.f32 {self._reg(x2)}, {x}, {x};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(om)}, {self._reg(x2)}, 0fBF800000, 0f3F800000;')  # 1-x^2
                    self._lines.append(f'    sqrt.approx.f32 {self._reg(sqr)}, {self._reg(om)};')
                    self._lines.append(f'    div.approx.f32 {self._reg(rat)}, {x}, {self._reg(sqr)};')
                    # atan of ratio
                    self._lines.append(f'    mul.f32 {self._reg(ax2)}, {self._reg(rat)}, {self._reg(rat)};')
                    self._lines.append(f'    mul.f32 {self._reg(ax4)}, {self._reg(ax2)}, {self._reg(ax2)};')
                    self._lines.append(f'    mul.f32 {self._reg(ax6)}, {self._reg(ax4)}, {self._reg(ax2)};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(ax6)}, 0fBE124925, 0f3F800000;')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p2)}, {self._reg(ax4)}, 0f3E4CCCCD, {self._reg(p)};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(ax2)}, 0fBEAAAAAB, {self._reg(p2)};')
                    self._lines.append(f'    mul.f32 {dest}, {self._reg(rat)}, {self._reg(p)};')
                elif inst.func in ('acosf', 'acos'):
                    # acos(x) = pi/2 - asin(x)
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    x2   = kernel.new_value(f'_acos_x2_{n}', FLOAT)
                    om   = kernel.new_value(f'_acos_om_{n}', FLOAT)
                    sqr  = kernel.new_value(f'_acos_sq_{n}', FLOAT)
                    rat  = kernel.new_value(f'_acos_r_{n}',  FLOAT)
                    ax2  = kernel.new_value(f'_acos_ax2_{n}', FLOAT)
                    ax4  = kernel.new_value(f'_acos_ax4_{n}', FLOAT)
                    ax6  = kernel.new_value(f'_acos_ax6_{n}', FLOAT)
                    p    = kernel.new_value(f'_acos_p_{n}',   FLOAT)
                    p2   = kernel.new_value(f'_acos_p2_{n}',  FLOAT)
                    asin_v = kernel.new_value(f'_acos_as_{n}', FLOAT)
                    self._lines.append(f'    mul.f32 {self._reg(x2)}, {x}, {x};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(om)}, {self._reg(x2)}, 0fBF800000, 0f3F800000;')
                    self._lines.append(f'    sqrt.approx.f32 {self._reg(sqr)}, {self._reg(om)};')
                    self._lines.append(f'    div.approx.f32 {self._reg(rat)}, {x}, {self._reg(sqr)};')
                    self._lines.append(f'    mul.f32 {self._reg(ax2)}, {self._reg(rat)}, {self._reg(rat)};')
                    self._lines.append(f'    mul.f32 {self._reg(ax4)}, {self._reg(ax2)}, {self._reg(ax2)};')
                    self._lines.append(f'    mul.f32 {self._reg(ax6)}, {self._reg(ax4)}, {self._reg(ax2)};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(ax6)}, 0fBE124925, 0f3F800000;')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p2)}, {self._reg(ax4)}, 0f3E4CCCCD, {self._reg(p)};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(p)}, {self._reg(ax2)}, 0fBEAAAAAB, {self._reg(p2)};')
                    self._lines.append(f'    mul.f32 {self._reg(asin_v)}, {self._reg(rat)}, {self._reg(p)};')
                    self._lines.append(f'    sub.f32 {dest}, 0f3FC90FDB, {self._reg(asin_v)};')  # pi/2 - asin
                elif inst.func in ('powf', 'pow'):
                    # powf(x, y) = exp2(y * log2(x)) — approx via f32 lg2/ex2
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    y = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f3F800000'
                    n = inst.dest.id if inst.dest else 0
                    lg2x = kernel.new_value(f'_pow_lg2x_{n}', FLOAT)
                    ylg2x = kernel.new_value(f'_pow_ylg2x_{n}', FLOAT)
                    self._lines.append(f'    lg2.approx.f32 {self._reg(lg2x)}, {x};')
                    self._lines.append(f'    mul.f32 {self._reg(ylg2x)}, {y}, {self._reg(lg2x)};')
                    self._lines.append(f'    ex2.approx.f32 {dest}, {self._reg(ylg2x)};')
                elif inst.func in ('hypotf', 'hypot'):
                    # hypotf(a, b) = sqrt(a*a + b*b)
                    a = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    a2 = kernel.new_value(f'_hypot_a2_{n}', FLOAT)
                    ab2 = kernel.new_value(f'_hypot_ab2_{n}', FLOAT)
                    self._lines.append(f'    mul.f32 {self._reg(a2)}, {a}, {a};')
                    self._lines.append(f'    fma.rn.f32 {self._reg(ab2)}, {b}, {b}, {self._reg(a2)};')
                    self._lines.append(f'    sqrt.approx.f32 {dest}, {self._reg(ab2)};')
                elif inst.func in ('cbrtf', 'cbrt'):
                    # cbrtf(x) = exp2(log2(x) / 3) — only valid for x > 0
                    x = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    n = inst.dest.id if inst.dest else 0
                    lg2x = kernel.new_value(f'_cbrt_lg2x_{n}', FLOAT)
                    lg2x3 = kernel.new_value(f'_cbrt_lg2x3_{n}', FLOAT)
                    self._lines.append(f'    lg2.approx.f32 {self._reg(lg2x)}, {x};')
                    self._lines.append(f'    mul.f32 {self._reg(lg2x3)}, {self._reg(lg2x)}, 0f3EAAAAAB;')  # 1/3
                    self._lines.append(f'    ex2.approx.f32 {dest}, {self._reg(lg2x3)};')
                elif inst.func in ('atan2f', 'atan2'):
                    # atan2f(y, x): use identity atan2(y,x) = atan(y/x) adjusted for quadrant.
                    # Approximation: atan(t) ≈ t*(pi/4 - (|t|-1)*(0.2447+0.0663*|t|)) for |t|<=1
                    # Full implementation: atan2 via sinf/cosf angle not feasible in PTX.
                    # Use: atan(y/x) with quadrant correction via pi constant.
                    y_op = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    x_op = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f3F800000'
                    n = inst.dest.id if inst.dest else 0
                    ratio  = kernel.new_value(f'_atan2_r_{n}', FLOAT)
                    lg2r   = kernel.new_value(f'_atan2_l_{n}', FLOAT)
                    ex2r   = kernel.new_value(f'_atan2_e_{n}', FLOAT)
                    sinv   = kernel.new_value(f'_atan2_sv_{n}', FLOAT)
                    cosv   = kernel.new_value(f'_atan2_cv_{n}', FLOAT)
                    # Use atan(y/x) via angle: sin(atan(t)) = t/sqrt(1+t^2), cos = 1/sqrt(1+t^2)
                    # Simplest usable approximation in PTX: sin/cos of atan can be computed as
                    # sin.approx(atan2_angle) — but atan itself has no PTX opcode.
                    # Workaround: emit atan(y/x) via a polynomial approximation in PTX.
                    # For now: compute angle = atan(y/x) using sin≈y*rsqrt(x²+y²), cos≈x*rsqrt(x²+y²)
                    # then angle ≈ atan2(sin,cos) — this is circular.
                    # Practical approximation: use div + polynomial atan approx.
                    # atan(t) ≈ (pi/4)*t - t*(|t|-1)*(0.2447 + 0.0663*|t|) for |t| <= 1
                    # Use a crude version: atan(t) ≈ (pi/4)*t for small t, scaled for atan2.
                    # Best practical option: ratio = y/x, result = atan_approx(ratio).
                    # We use a 5-instruction polynomial: good enough for correctness probing.
                    t2     = kernel.new_value(f'_atan2_t2_{n}', FLOAT)
                    t4     = kernel.new_value(f'_atan2_t4_{n}', FLOAT)
                    poly   = kernel.new_value(f'_atan2_p_{n}', FLOAT)
                    self._lines.append(f'    div.approx.f32 {self._reg(ratio)}, {y_op}, {x_op};')
                    self._lines.append(f'    mul.f32 {self._reg(t2)}, {self._reg(ratio)}, {self._reg(ratio)};')
                    self._lines.append(f'    mul.f32 {self._reg(t4)}, {self._reg(t2)}, {self._reg(t2)};')
                    # poly = ratio * (1 - t2*0.3333 + t4*0.2)  — Maclaurin truncated
                    self._lines.append(f'    fma.rn.f32 {self._reg(poly)}, {self._reg(t4)}, 0f3E4CCCCD, {self._reg(ratio)};')  # +t4*0.2
                    self._lines.append(f'    fma.rn.f32 {dest}, {self._reg(t2)}, 0fBEAAAAAB, {self._reg(poly)};')              # -t2*0.333

    def _emit_term(self, term):
        if isinstance(term, RetTerm):
            self._lines.append('    ret;')
        elif isinstance(term, BrTerm):
            self._lines.append(f'    bra {term.target};')
        elif isinstance(term, CondBrTerm):
            if isinstance(term.cond, Value):
                if term.cond.id in self._pred_ids:
                    pred = self._reg(term.cond)
                else:
                    # Integer condition — synthesize a predicate register
                    p_idx = self._reg_counts.get('p', 0)
                    self._reg_counts['p'] = p_idx + 1
                    pred = f'%p{p_idx}'
                    ptx_ty = _ptx_type(term.cond.ty)
                    src = self._reg(term.cond)
                    self._lines.append(
                        f'    setp.ne.{ptx_ty} {pred}, {src}, 0;')
                self._lines.append(f'    @{pred} bra {term.true_bb};')
                self._lines.append(f'    bra {term.false_bb};')
            elif isinstance(term.cond, Const) and term.cond.value != 0:
                # Constant-true condition (e.g. for(;;)): unconditional branch
                self._lines.append(f'    bra {term.true_bb};')
            else:
                # Constant-false (unreachable true branch) or other constant
                self._lines.append(f'    bra {term.false_bb};')


def ir_to_ptx(module: Module) -> dict[str, str]:
    """Convert IR module to PTX text for each kernel."""
    emitter = PTXEmitter()
    result = {}
    for kernel in module.kernels:
        result[kernel.name] = emitter.emit_kernel(kernel)

    # Emit module-level global/constant variable declarations
    if module.global_vars:
        decl_lines = []
        for (sym_name, elem_ty, count, addr_space) in module.global_vars:
            space = 'const' if addr_space == AddrSpace.CONST else 'global'
            if isinstance(elem_ty, StructTy):
                # Struct global: allocate raw bytes for the full struct (× count).
                # Using .u32 would only allocate 4 bytes regardless of struct size.
                struct_align = max(
                    (ft.size for _, ft in elem_ty.fields if hasattr(ft, 'size')),
                    default=4)
                sa = 1
                while sa < struct_align and sa < 16:
                    sa <<= 1
                total_bytes = elem_ty.size * count
                decl_lines.append(
                    f'.visible .{space} .align {sa} .b8 {sym_name}[{total_bytes}];')
            else:
                ptx_ty = _ptx_type(elem_ty)
                align = elem_ty.size if isinstance(elem_ty, ScalarTy) else 8
                if count > 1:
                    decl_lines.append(
                        f'.visible .{space} .align {align} .{ptx_ty} {sym_name}[{count}];')
                else:
                    decl_lines.append(
                        f'.visible .{space} .align {align} .{ptx_ty} {sym_name};')
        emitter._module_preamble = decl_lines + emitter._module_preamble

    # Collect module-level preamble (global var decls, printf globals, vprintf extern)
    if emitter._module_preamble:
        result['__preamble__'] = '\n'.join(emitter._module_preamble)

    return result
