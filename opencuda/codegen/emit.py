"""
OpenCUDA codegen — lower IR to PTX text, then compile via OpenPTXas.

Strategy: IR → PTX text → OpenPTXas pipeline → cubin.
This reuses OpenPTXas's full backend (parser, regalloc, isel, scoreboard, emitter).
"""

from __future__ import annotations
import struct
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, PrintfInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)
from ..ir.types import (Type, ScalarTy, PtrTy, ScalarType, AddrSpace,
                         INT32, UINT32, FLOAT, VOID, DOUBLE)


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
        if isinstance(inst, (LoadInst,)):
            _note_def(inst.dest, i)
        if isinstance(inst, (CallInst,)):
            if inst.dest:
                _note_def(inst.dest, i)
        if isinstance(inst, (ParamInst,)):
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
        self._base_fallback_id: int = 0
        self._pred_ids: set[int] = set()
        self._val_type_map: dict[int, Type] = {}

    def _reg(self, v: Value) -> str:
        prefix = 'p' if v.id in self._pred_ids else _ptx_reg_prefix(v.ty)
        if v.id in self._alloc:
            phys = self._alloc[v.id]
        else:
            # Fallback for values created during emission (e.g., widen temps)
            max_alloc = self._alloc_max.get(prefix, -1) + 1
            phys = max_alloc + (v.id - self._base_fallback_id)
        self._reg_counts[prefix] = max(self._reg_counts.get(prefix, 0), phys + 1)
        return f'%{prefix}{phys}'

    def _operand(self, op: Operand, force_type: str = None) -> str:
        if isinstance(op, Value):
            return self._reg(op)
        if isinstance(op, Const):
            is_fp = force_type in ('f32', 'f64') if force_type else (
                isinstance(op.ty, ScalarTy) and op.ty.is_float)
            is_half = force_type == 'f16'
            if is_half:
                return f'0h{_half_hex(float(op.value))}'
            if is_fp:
                return f'0f{self._float_hex(float(op.value))}'
            return str(int(op.value))
        return str(op)

    def _float_hex(self, f: float) -> str:
        return struct.pack('>f', f).hex().upper()

    def emit_kernel(self, kernel: Kernel) -> str:
        self._lines = []
        self._reg_counts = {}
        self._shared_val_ids: dict[str, list] = {}
        self._widen_cache = {}
        self._printf_call_count = 0

        # Run linear scan allocation
        self._alloc, self._val_type_map, self._pred_ids, self._alloc_max = _build_alloc_map(kernel)
        self._base_fallback_id = kernel._next_id

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

        # Kernel signature
        params = ', '.join(
            f'.param .{_ptx_type(p.ty)} {p.name}' for p in kernel.params
        )
        ptx.append(f'.visible .entry {kernel.name}(')
        ptx.append(f'    {params})')
        ptx.append('{')

        # Shared memory declarations
        if hasattr(kernel, '_shared_decls'):
            for sname, sty, scount in kernel._shared_decls:
                ptx_sty = _ptx_type(sty)
                ptx.append(f'    .shared .{ptx_sty} {sname}[{scount}];')

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

        # Initialize shared memory base addresses
        if hasattr(kernel, '_shared_decls'):
            smem_inits = []
            for sname, sty, scount in kernel._shared_decls:
                for val in self._shared_val_ids.get(sname, []):
                    reg = self._reg(val)
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
        if isinstance(inst, ParamInst):
            ty = kernel.params[inst.param_index].ty
            ptx_ty = _ptx_type(ty)
            self._lines.append(
                f'    ld.param.{ptx_ty} {self._reg(inst.dest)}, [{inst.param_name}];')

        elif isinstance(inst, BinInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
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
            # Half precision arithmetic
            if _is_half(ty):
                half_op_map = {
                    BinOp.ADD: 'add', BinOp.SUB: 'sub',
                    BinOp.MUL: 'mul', BinOp.DIV: 'div.approx',
                }
                ptx_op = half_op_map.get(inst.op, 'add')
                self._lines.append(
                    f'    {ptx_op}.f16 {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, "f16")}, {self._operand(inst.rhs, "f16")};')
                return
            # Bitwise ops use .b32 type, not .s32/.u32
            if inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR, BinOp.SHL, BinOp.SHR):
                ptx_ty = f'b{ty.size * 8}' if isinstance(ty, ScalarTy) else 'b32'

            # Pointer arithmetic: use u64 for add/sub
            if _is_ptr(ty) and inst.op in (BinOp.ADD, BinOp.SUB):
                lhs = self._operand(inst.lhs)
                rhs = self._operand(inst.rhs)
                if isinstance(inst.rhs, (Value, Const)) and not _is_64bit(inst.rhs.ty if isinstance(inst.rhs, Value) else INT32):
                    rhs_id = inst.rhs.id if isinstance(inst.rhs, Value) else id(inst.rhs)
                    if rhs_id in self._widen_cache:
                        wide = self._widen_cache[rhs_id]
                    else:
                        wide = kernel.new_value(f'wide{inst.dest.id}', ty)
                        self._reg_counts[_ptx_reg_prefix(ty)] = max(
                            self._reg_counts.get(_ptx_reg_prefix(ty), 0), wide.id + 1)
                        self._lines.append(
                            f'    cvt.u64.u32 {self._reg(wide)}, {rhs};')
                        self._widen_cache[rhs_id] = wide
                    rhs = self._reg(wide)
                self._lines.append(
                    f'    {ptx_op}.u64 {self._reg(inst.dest)}, {lhs}, {rhs};')
            elif _is_float(ty):
                fty = _ptx_type(ty)  # f32 or f64
                self._lines.append(
                    f'    {ptx_op}.{fty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, fty)}, {self._operand(inst.rhs, fty)};')
            elif _is_64bit(ty):
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')
            else:
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')

        elif isinstance(inst, CmpInst):
            ty = inst.lhs.ty if isinstance(inst.lhs, Value) else INT32
            ptx_ty = _ptx_type(ty)
            op_map = {
                CmpOp.LT: 'lt', CmpOp.LE: 'le', CmpOp.GT: 'gt',
                CmpOp.GE: 'ge', CmpOp.EQ: 'eq', CmpOp.NE: 'ne',
            }
            cmp_str = op_map[inst.op]
            pred = self._reg(inst.dest)
            self._lines.append(
                f'    setp.{cmp_str}.{ptx_ty} {pred}, '
                f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')

        elif isinstance(inst, CvtInst):
            src_ty = inst.src.ty if isinstance(inst.src, Value) else INT32
            dst_ty = inst.dest.ty
            src_ptx = _ptx_type(src_ty)
            dst_ptx = _ptx_type(dst_ty)
            rnd = _cvt_modifier(dst_ty, src_ty)
            self._lines.append(
                f'    cvt{rnd}.{dst_ptx}.{src_ptx} {self._reg(inst.dest)}, {self._operand(inst.src)};')

        elif isinstance(inst, LoadInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            nc = False
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
                elif inst.addr.ty.addr_space == AddrSpace.CONST:
                    addr_space = 'global'
                    nc = True
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
            ty = inst.value.ty if isinstance(inst.value, Value) else INT32
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
            self._lines.append(
                f'    st.{addr_space}.{ptx_ty} [{self._operand(inst.addr)}], '
                f'{self._operand(inst.value, ptx_ty)};')

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
                            f'    cvt.rn.f64.f32 {self._reg(promoted)}, {self._operand(arg)};')
                        store_val = self._reg(promoted)
                        store_ty = 'f64'
                    elif not _is_64bit(arg_ty) and not _is_float(arg_ty):
                        # Widen int32 → u64
                        widened = kernel.new_value(f'_va_arg_{n}_{i}', PtrTy(VOID, AddrSpace.GLOBAL))
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
                    'atomicAdd': 'add', 'atomicSub': 'add',
                    'atomicMin': 'min', 'atomicMax': 'max',
                    'atomicAnd': 'and', 'atomicOr': 'or', 'atomicXor': 'xor',
                    'atomicExch': 'exch', 'atomicCAS': 'cas',
                }
                ptx_op = atomic_ops.get(inst.func, 'add')
                addr = self._operand(inst.args[0]) if inst.args else '%rd0'
                val_ty = 'u32'
                if len(inst.args) > 1 and isinstance(inst.args[1], Value):
                    val_ty = _ptx_type(inst.args[1].ty)
                elif len(inst.args) > 1 and isinstance(inst.args[1], Const):
                    if isinstance(inst.args[1].value, float):
                        val_ty = 'f32'
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                self._lines.append(
                    f'    atom.global.{ptx_op}.{val_ty} {dest}, [{addr}], {self._operand(inst.args[1], val_ty)};')
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
            elif inst.func == '__ballot_sync':
                mask = self._operand(inst.args[0]) if inst.args else '0xFFFFFFFF'
                pred_val = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                self._lines.append(
                    f'    vote.sync.ballot.b32 {dest}, {pred_val}, {mask};')
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

    def _emit_term(self, term):
        if isinstance(term, RetTerm):
            self._lines.append('    ret;')
        elif isinstance(term, BrTerm):
            self._lines.append(f'    bra {term.target};')
        elif isinstance(term, CondBrTerm):
            if isinstance(term.cond, Value):
                pred = self._reg(term.cond)
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

    # Collect module-level preamble (printf globals, vprintf extern)
    if emitter._module_preamble:
        result['__preamble__'] = '\n'.join(emitter._module_preamble)

    return result
