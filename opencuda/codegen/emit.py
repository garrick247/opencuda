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
from ..ir.types import (Type, ScalarTy, PtrTy, ScalarType, AddrSpace,
                         INT32, UINT32, INT64, UINT64, FLOAT, VOID, DOUBLE, HALF)


def _ptx_type(ty: Type) -> str:
    """Convert IR type to PTX type string."""
    if isinstance(ty, PtrTy):
        return 'u64'
    if isinstance(ty, ScalarTy):
        mapping = {
            ScalarType.VOID: 'u32',
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
            live_end[op.id] = idx

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

    back_edges: list[tuple[int, int]] = []  # (loop_header_start, back_edge_end)
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
                back_edges.append((bb_flat_start[tgt], bb_flat_end[bb.label]))

    for header_start, loop_end in back_edges:
        for val_id, le in list(live_end.items()):
            ls = live_start.get(val_id, 0)
            if ls <= header_start and header_start <= le <= loop_end:
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

        # Pre-scan: find Values that are shared memory variables
        if hasattr(kernel, '_shared_decls'):
            smem_names = {s[0] for s in kernel._shared_decls}
            for bb in kernel.blocks:
                for inst in bb.instructions:
                    if hasattr(inst, 'lhs') and isinstance(inst.lhs, Value):
                        if inst.lhs.name in smem_names:
                            self._shared_val_ids.setdefault(inst.lhs.name, []).append(inst.lhs)
                    if hasattr(inst, 'dest') and isinstance(inst.dest, Value):
                        if inst.dest.name in smem_names:
                            self._shared_val_ids.setdefault(inst.dest.name, []).append(inst.dest)

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
                    ptx.append(f'    .shared .{ptx_sty} {sname}[{scount}];')

        # Local memory (stack) array declarations
        if hasattr(kernel, '_local_decls'):
            for lname, lty, lcount, _val in kernel._local_decls:
                elem_bytes = lty.size if hasattr(lty, 'size') else 4
                total_bytes = elem_bytes * lcount
                # PTX .align must be a power of two: round elem_bytes up
                align = 1
                while align < elem_bytes:
                    align <<= 1
                align = min(align, 16)  # PTX max useful alignment
                ptx.append(f'    .local .align {align} .b8 {lname}_local[{total_bytes}];')

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
            for lname, _lty, _lcount, lval in kernel._local_decls:
                reg = self._reg(lval)
                if reg not in emitted_local_regs:
                    emitted_local_regs.add(reg)
                    local_inits.append(f'    mov.u64 {reg}, {lname}_local;')
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
            if inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR, BinOp.SHL):
                ptx_ty = f'b{ty.size * 8}' if isinstance(ty, ScalarTy) else 'b32'
            elif inst.op == BinOp.SHR:
                if isinstance(ty, ScalarTy) and ty.is_signed:
                    ptx_ty = f's{ty.size * 8}'  # arithmetic right shift
                else:
                    ptx_ty = f'b{ty.size * 8}' if isinstance(ty, ScalarTy) else 'b32'

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
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty, kernel)}, '
                    f'{self._operand(inst.rhs, ptx_ty, kernel)};')
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
            op_map = {
                CmpOp.LT: 'lt', CmpOp.LE: 'le', CmpOp.GT: 'gt',
                CmpOp.GE: 'ge', CmpOp.EQ: 'eq', CmpOp.NE: 'ne',
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
            src_ptx = _ptx_type(src_ty)
            dst_ptx = _ptx_type(dst_ty)
            rnd = _cvt_modifier(dst_ty, src_ty)
            # Predicate → integer: emit selp first, then cvt on the selp result
            src_op = self._operand(inst.src, src_ptx, kernel)
            self._lines.append(
                f'    cvt{rnd}.{dst_ptx}.{src_ptx} {self._reg(inst.dest)}, {src_op};')

        elif isinstance(inst, LoadInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            nc = False
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
                elif inst.addr.ty.addr_space == AddrSpace.LOCAL:
                    addr_space = 'local'
                elif inst.addr.ty.addr_space == AddrSpace.CONST:
                    addr_space = 'global'
                    nc = True
            elif isinstance(inst.addr, SymbolRef) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.CONST:
                    addr_space = 'const'
                elif inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
            # PTX does not support ld.f16 — use b16 instead
            if ptx_ty == 'f16':
                ptx_ty = 'b16'
            if nc:
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
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
                elif inst.addr.ty.addr_space == AddrSpace.LOCAL:
                    addr_space = 'local'
            # Predicate registers cannot be stored directly — convert to 0/1 integer first.
            if isinstance(inst.value, Value) and inst.value.id in self._pred_ids:
                val_str = self._pred_to_int(inst.value, ptx_ty, kernel)
            else:
                val_str = self._operand(inst.value, ptx_ty)
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
                    arg_ty = arg.ty if isinstance(arg, Value) else INT32
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
                    # atomicCAS(addr, compare, val) — 3-arg: PTX atom.cas.b32
                    # atom.{space}.cas.b32 dest, [addr], compare, val;
                    cmp_arg = inst.args[1] if len(inst.args) > 1 else Const(INT32, 0)
                    new_arg = inst.args[2] if len(inst.args) > 2 else Const(INT32, 0)
                    # CAS uses b32 type (bitwise comparison)
                    self._lines.append(
                        f'    atom.{_atom_space}.cas.b32 {dest}, [{addr}], '
                        f'{self._operand(cmp_arg, "b32")}, {self._operand(new_arg, "b32")};')
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
                }
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
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                if inst.func in ('__threadfence',):
                    # Global memory fence (visible to all threads)
                    self._lines.append('    membar.gl;')
                elif inst.func in ('__threadfence_block',):
                    # Block-level memory fence
                    self._lines.append('    membar.cta;')
                elif inst.func in ('__syncwarp',):
                    # Warp-level synchronization
                    mask = self._operand(inst.args[0]) if inst.args else '0xffffffff'
                    self._lines.append(f'    bar.warp.sync {mask};')
                elif inst.func in ('__activemask',):
                    # Returns bitmask of active lanes in warp
                    self._lines.append(f'    activemask.b32 {dest};')
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
                    ptx_op = _f32_binary[inst.func]
                    a = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    self._lines.append(f'    {ptx_op} {dest}, {a}, {b};')
                elif inst.func in ('fmaf', 'fma'):
                    # Fused multiply-add: fma(a, b, c) = a*b + c (single rounding)
                    a = self._operand(inst.args[0]) if inst.args else '0f00000000'
                    b = self._operand(inst.args[1]) if len(inst.args) > 1 else '0f00000000'
                    c = self._operand(inst.args[2]) if len(inst.args) > 2 else '0f00000000'
                    self._lines.append(f'    fma.rn.f32 {dest}, {a}, {b}, {c};')
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
            else:
                pred = '%p0'
            self._lines.append(f'    @{pred} bra {term.true_bb};')
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
            ptx_ty = _ptx_type(elem_ty)
            align = elem_ty.size if isinstance(elem_ty, ScalarTy) else 8
            space = 'const' if addr_space == AddrSpace.CONST else 'global'
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
