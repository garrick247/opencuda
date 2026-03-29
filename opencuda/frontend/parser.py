"""
OpenCUDA parser — CUDA-subset C to IR.

Supported:
  - __global__ kernel functions
  - Types: int, unsigned int, float, void, pointers
  - Expressions: arithmetic, comparison, array indexing, member access
  - Statements: variable decl, assignment, if/else, for, return
  - Built-ins: threadIdx.x/y/z, blockIdx.x/y/z, blockDim.x/y/z, __syncthreads()
"""

from __future__ import annotations
from typing import Optional

from .lexer import Token, TokKind, lex  # noqa: F401 (TokKind members used throughout)
from ..ir.types import (Type, ScalarTy, PtrTy, AddrSpace, ScalarType, StructTy,
                         INT32, UINT32, FLOAT, VOID, INT64, UINT64, DOUBLE, HALF)
from ..ir.nodes import (Module, Kernel, KernelParam, BasicBlock,
                         Value, Const, Operand, SymbolRef, GlobalAddrInst,
                         BinInst, CmpInst, LoadInst, StoreInst,
                         CvtInst, CallInst, ParamInst, PrintfInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self._toks = tokens
        self._pos = 0
        self._kernel: Optional[Kernel] = None
        self._cur_block: Optional[BasicBlock] = None
        self._variables: dict[str, Value] = {}
        self._block_count = 0
        self._struct_types: dict[str, StructTy] = {}  # struct name → StructTy
        self._typedefs: dict[str, Type] = {}  # typedef name → Type
        # For struct array members: maps struct_name → {field_base_name → array_count}
        # e.g. Vec4 with "float data[4]" → {'Vec4': {'data': 4}}
        self._struct_field_arrays: dict[str, dict[str, int]] = {}
        # Pre-register CUDA built-in vector types (float2, int3, etc.)
        self._register_builtin_vector_types()
        self._break_targets: list[str] = []          # stack of break target labels
        self._break_snapshots: list[Optional[dict]] = []  # vars snapshot at break scope entry
        self._continue_targets: list[str] = []       # stack of continue target labels
        self._inline_return_target = None  # (return_dest_val, return_merge_label) or None
        # Multi-dimensional array row strides: maps var_name → row_stride_bytes.
        # For float tile[16][16], row_stride = 16*sizeof(float) = 64.
        # Used to compute tile[i][j] as *(tile + i*64 + j*4) rather than *(tile + i*4 + j*4).
        self._array_row_strides: dict[str, int] = {}
        # Module-level compile-time constants (enum values, etc.)
        # These are visible in all kernels as Const operands without IR instructions.
        self._global_consts: dict[str, Const] = {}
        # Module-level extern __shared__ declarations: list of (name, scalar_ty).
        # Injected into each kernel's scope at kernel parse time.
        self._module_shared_decls: list = []
        # Register C/CUDA scalar aliases not covered by keyword tokens
        self._typedefs['bool'] = INT32        # _Bool / bool → i32 (0 or 1)
        self._typedefs['size_t'] = UINT64     # pointer-sized unsigned
        self._typedefs['ptrdiff_t'] = INT64
        self._typedefs['intptr_t'] = INT64
        self._typedefs['uintptr_t'] = UINT64
        self._typedefs['int8_t'] = INT32
        self._typedefs['int16_t'] = INT32
        self._typedefs['int32_t'] = INT32
        self._typedefs['int64_t'] = INT64
        self._typedefs['uint8_t'] = UINT32
        self._typedefs['uint16_t'] = UINT32
        self._typedefs['uint32_t'] = UINT32
        self._typedefs['uint64_t'] = UINT64
        self._global_consts['true'] = Const(INT32, 1)
        self._global_consts['false'] = Const(INT32, 0)
        self._global_consts['NULL'] = Const(UINT64, 0)
        self._global_consts['nullptr'] = Const(UINT64, 0)

    def _register_builtin_vector_types(self):
        """Pre-register CUDA built-in vector types as StructTy.

        float2/float3/float4, int2/int3/int4, uint2/uint3/uint4,
        double2, char2/char4, short2/short4, etc.
        These are accessed via .x/.y/.z/.w member syntax.
        """
        def _vec(base_ty, names):
            fields = tuple((n, base_ty) for n in names)
            st = StructTy('__vec__', fields)
            return st

        _xy = ('x', 'y')
        _xyz = ('x', 'y', 'z')
        _xyzw = ('x', 'y', 'z', 'w')

        for typename, base_ty, members in [
            ('float2', FLOAT, _xy),   ('float3', FLOAT, _xyz),  ('float4', FLOAT, _xyzw),
            ('double2', DOUBLE, _xy),
            ('int2', INT32, _xy),     ('int3', INT32, _xyz),    ('int4', INT32, _xyzw),
            ('uint2', UINT32, _xy),   ('uint3', UINT32, _xyz),  ('uint4', UINT32, _xyzw),
            ('short2', INT32, _xy),   ('ushort2', UINT32, _xy),
            ('char2', INT32, _xy),    ('uchar2', UINT32, _xy),
            ('char4', INT32, _xyzw),  ('uchar4', UINT32, _xyzw),
            ('long2', INT64, _xy),    ('ulong2', UINT64, _xy),
            ('longlong2', INT64, _xy), ('ulonglong2', UINT64, _xy),
        ]:
            fields = tuple((n, base_ty) for n in members)
            st = StructTy(typename, fields)
            self._struct_types[typename] = st
            self._typedefs[typename] = st

    # -- Token helpers -------------------------------------------------------

    def _peek(self) -> Token:
        return self._toks[self._pos]

    def _advance(self) -> Token:
        tok = self._toks[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: TokKind) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise ParseError(f"Line {tok.line}: expected {kind.name}, got {tok.kind.name} '{tok.value}'")
        return self._advance()

    def _match(self, kind: TokKind) -> Optional[Token]:
        if self._peek().kind == kind:
            return self._advance()
        return None

    def _at(self, kind: TokKind) -> bool:
        return self._peek().kind == kind

    # -- IR helpers ----------------------------------------------------------

    def _new_block(self, prefix: str = "BB") -> BasicBlock:
        self._block_count += 1
        label = f"{prefix}_{self._block_count}"
        bb = BasicBlock(label=label)
        self._kernel.blocks.append(bb)
        return bb

    def _emit(self, inst):
        self._cur_block.instructions.append(inst)

    def _new_val(self, name: str, ty: Type) -> Value:
        return self._kernel.new_value(name, ty)

    @staticmethod
    def _const_fold(op: 'BinOp', lhs: Operand, rhs: Operand) -> 'Const | None':
        """Fold two compile-time constants into a single Const. Returns None if
        either operand is not Const or the operation is not constant-foldable."""
        if not isinstance(lhs, Const) or not isinstance(rhs, Const):
            return None
        a, b = int(lhs.value), int(rhs.value)
        ty = lhs.ty if isinstance(lhs.ty, ScalarTy) else INT32
        try:
            result = {
                BinOp.ADD: a + b,
                BinOp.SUB: a - b,
                BinOp.MUL: a * b,
                BinOp.DIV: a // b if b != 0 else 0,
                BinOp.MOD: a % b if b != 0 else 0,
                BinOp.OR:  a | b,
                BinOp.AND: a & b,
                BinOp.XOR: a ^ b,
                BinOp.SHL: a << (b & 63),
                BinOp.SHR: a >> (b & 63),
            }.get(op)
        except Exception:
            return None
        if result is None:
            return None
        return Const(ty, result)

    def _loop_writeback(self, entry_vars: dict) -> None:
        """Write modified loop variables back to their canonical entry Values.

        For each variable that was assigned inside a loop body, emit:
            add <entry_val>, <current_val>, 0
        in the current block, then restore _variables to the canonical entry
        Values so that the loop condition (which was built from entry_vals)
        sees the updated values on re-entry.

        This mirrors the for-loop writeback in inc_bb, applied to while/do-while.
        NOTE: continue statements bypass this writeback — they branch directly to
        cond_bb without running through the end of body. That is a known
        limitation: mutating the loop-condition variable and using continue in
        the same while loop may produce incorrect loop termination.
        """
        for var_name, entry_val in entry_vars.items():
            if not isinstance(entry_val, Value):
                continue
            cur_val = self._variables.get(var_name)
            if cur_val is None or cur_val is entry_val:
                continue
            if isinstance(cur_val, Value):
                self._emit(BinInst(entry_val, BinOp.ADD, cur_val, Const(entry_val.ty, 0)))
                self._variables[var_name] = entry_val
            elif isinstance(cur_val, Const):
                # Constant assigned to a canonical register: emit copy so that
                # constant_fold can materialise the value into entry_val.
                self._emit(BinInst(entry_val, BinOp.ADD, cur_val, Const(entry_val.ty, 0)))
                self._variables[var_name] = entry_val

    # -- Type parsing --------------------------------------------------------

    def _parse_type(self) -> Type:
        """Parse a C type specifier."""
        tok = self._peek()

        if tok.kind == TokKind.KW_VOID:
            self._advance()
            return VOID
        elif tok.kind == TokKind.KW_FLOAT:
            self._advance()
            return FLOAT
        elif tok.kind == TokKind.KW_DOUBLE:
            self._advance()
            return DOUBLE
        elif tok.kind == TokKind.KW_HALF:
            self._advance()
            return HALF
        elif tok.kind == TokKind.KW_INT:
            self._advance()
            return INT32
        elif tok.kind == TokKind.KW_UNSIGNED:
            self._advance()
            if self._match(TokKind.KW_INT):
                pass
            elif self._match(TokKind.KW_SHORT):
                self._match(TokKind.KW_INT)
            elif self._match(TokKind.KW_CHAR):
                pass
            elif self._match(TokKind.KW_LONG):
                if self._match(TokKind.KW_LONG):
                    return UINT64
            return UINT32
        elif tok.kind == TokKind.KW_LONG:
            self._advance()
            if self._match(TokKind.KW_LONG):
                return INT64
            return INT32  # treat 'long' as int32 for simplicity
        elif tok.kind == TokKind.KW_SHORT:
            self._advance()
            self._match(TokKind.KW_INT)  # optional trailing 'int'
            return INT32  # treat 'short' as int32 (no sub-word PTX registers)
        elif tok.kind == TokKind.KW_SIGNED:
            self._advance()
            # signed [int | char | long | short] — consume optional base type
            self._match(TokKind.KW_INT) or self._match(TokKind.KW_CHAR) \
                or self._match(TokKind.KW_SHORT) or self._match(TokKind.KW_LONG)
            return INT32
        elif tok.kind == TokKind.KW_CHAR:
            self._advance()
            return INT32  # treat 'char' as int32 for simplicity
        elif tok.kind == TokKind.KW_BOOL:
            self._advance()
            return INT32  # treat 'bool' as int32

        # Struct or union type
        if tok.kind in (TokKind.KW_STRUCT, TokKind.KW_UNION):
            kw = self._advance()
            sname = self._expect(TokKind.IDENT).value
            if sname in self._struct_types:
                return self._struct_types[sname]
            raise ParseError(f"Line {kw.line}: undefined struct/union '{sname}'")

        # Typedef'd type
        if tok.kind == TokKind.IDENT and tok.value in self._typedefs:
            self._advance()
            return self._typedefs[tok.value]

        # Bare struct name (C++ style: Vec3 instead of struct Vec3)
        if tok.kind == TokKind.IDENT and tok.value in self._struct_types:
            self._advance()
            return self._struct_types[tok.value]

        # Template type parameter (T, U, etc.) — treat as float32.
        # C++ templates are not fully supported; this allows parsing template
        # function bodies that use a type parameter as return/param type.
        if tok.kind == TokKind.IDENT and tok.value.isupper() and len(tok.value) <= 2:
            self._advance()
            return FLOAT

        raise ParseError(f"Line {tok.line}: expected type, got '{tok.value}'")

    def _parse_type_with_ptr(self) -> Type:
        """Parse type followed by optional pointer stars.

        const T * __restrict__ → AddrSpace.CONST (emits ld.global.nc).
        Both qualifiers must be present: const alone doesn't exclude aliasing,
        and __restrict__ alone doesn't make the data read-only.
        """
        # Track const on the pointee (before or immediately after the base type).
        # volatile/static/inline/register are silently consumed — no PTX semantics.
        self._match(TokKind.KW_STATIC)    # static/inline/register before type
        self._match(TokKind.KW_VOLATILE)  # volatile before type (e.g. volatile int *)
        pointee_const = self._match(TokKind.KW_CONST)
        self._match(TokKind.KW_VOLATILE)  # volatile after const (e.g. const volatile T *)
        self._match(TokKind.KW_STATIC)
        base = self._parse_type()
        # "float const *" or "float volatile *" — qualifier after base type
        if self._match(TokKind.KW_CONST):
            pointee_const = True
        self._match(TokKind.KW_VOLATILE)
        while self._match(TokKind.STAR):
            # "float * const" — const AFTER star means pointer-const, not pointee-const
            ptr_const = self._match(TokKind.KW_CONST)  # noqa: F841 (consumed, not used)
            self._match(TokKind.KW_VOLATILE)            # "float * volatile" — skip
            base = PtrTy(base, AddrSpace.GLOBAL)
        # __restrict__ qualifier
        has_restrict = (self._at(TokKind.IDENT) and self._peek().value == '__restrict__')
        if has_restrict:
            self._advance()
        # Upgrade to CONST addr space when programmer guarantees read-only + no aliasing.
        # This allows codegen to emit ld.global.nc (non-caching / read-only cache).
        if pointee_const and has_restrict and isinstance(base, PtrTy):
            base = PtrTy(base.pointee, AddrSpace.CONST)
        return base

    # -- Expression parsing (precedence climbing) ----------------------------

    def _parse_expr(self) -> Operand:
        return self._parse_assign_expr()

    def _parse_assign_expr(self) -> Operand:
        # Capture the original source-level variable name before parsing the LHS.
        # _parse_or_expr() resolves variable aliases (variables['val'] may return
        # Value('elem', ...)), so lhs.name would update the wrong key in _variables.
        _lhs_orig_name = None
        if self._at(TokKind.IDENT):
            cand = self._peek().value
            next_pos = self._pos + 1
            _assign_op_kinds = (
                TokKind.ASSIGN, TokKind.PLUS_EQ, TokKind.MINUS_EQ,
                TokKind.STAR_EQ, TokKind.SLASH_EQ, TokKind.PERCENT_EQ,
                TokKind.AMP_EQ, TokKind.PIPE_EQ, TokKind.CARET_EQ,
                TokKind.LSHIFT_EQ, TokKind.RSHIFT_EQ)
            if (next_pos < len(self._toks)
                    and self._toks[next_pos].kind in _assign_op_kinds
                    and cand in self._variables):
                _lhs_orig_name = cand
            elif (next_pos < len(self._toks)
                    and self._toks[next_pos].kind == TokKind.DOT
                    and cand in self._variables
                    and isinstance(self._variables[cand].ty, StructTy)):
                # struct.field = val — map lhs to the per-field variable so the
                # assignment updates the variable binding instead of emitting a
                # StoreInst through the field's pointer value.
                field_pos = next_pos + 1
                if field_pos < len(self._toks) and self._toks[field_pos].kind == TokKind.IDENT:
                    field_name = self._toks[field_pos].value
                    assign_pos = field_pos + 1
                    if (assign_pos < len(self._toks)
                            and self._toks[assign_pos].kind in _assign_op_kinds):
                        compound_name = f"{cand}_{field_name}"
                        if compound_name in self._variables:
                            _lhs_orig_name = compound_name

        lhs = self._parse_or_expr()
        # Ternary: cond ? true_expr : false_expr
        if self._match(TokKind.QUESTION):
            true_val = self._parse_expr()
            self._expect(TokKind.COLON)
            false_val = self._parse_assign_expr()
            # Lower to: if (cond) { dest = true } else { dest = false }
            # Use _result_type so that float constants (Const(FLOAT, 1.0)) are
            # considered — not just Value instances.
            result_ty = self._result_type(true_val, false_val)
            dest = self._new_val("ternary", result_ty)
            true_bb = self._new_block("tern_true")
            false_bb = self._new_block("tern_false")
            merge_bb = self._new_block("tern_merge")
            self._cur_block.terminator = CondBrTerm(lhs, true_bb.label, false_bb.label)
            self._cur_block = true_bb
            # Emit: dest = true_val
            self._emit(BinInst(dest, BinOp.ADD, true_val,
                               Const(result_ty, 0.0 if (isinstance(result_ty, ScalarTy) and result_ty.is_float) else 0)))
            true_bb.terminator = BrTerm(merge_bb.label)
            self._cur_block = false_bb
            self._emit(BinInst(dest, BinOp.ADD, false_val,
                               Const(result_ty, 0.0 if (isinstance(result_ty, ScalarTy) and result_ty.is_float) else 0)))
            false_bb.terminator = BrTerm(merge_bb.label)
            self._cur_block = merge_bb
            return dest
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_assign_expr()
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                # If _lhs_orig_name is set it means the LHS came from a struct
                # field access (struct.field = val). In that case update the
                # per-field variable binding — do NOT emit a StoreInst through
                # the pointer value, which would be semantically wrong.
                if _lhs_orig_name and _lhs_orig_name in self._variables:
                    self._variables[_lhs_orig_name] = rhs
                    return rhs
                # Coerce rhs to pointee type on scalar type mismatch (e.g. half → float*)
                if (isinstance(rhs, Value) and isinstance(rhs.ty, ScalarTy)
                        and isinstance(lhs.ty.pointee, ScalarTy)
                        and rhs.ty != lhs.ty.pointee):
                    coerced = self._new_val("coerce", lhs.ty.pointee)
                    self._emit(CvtInst(coerced, rhs))
                    rhs = coerced
                self._emit(StoreInst(addr=lhs, value=rhs))
                return rhs
            # Variable assignment: update the variable binding.
            # Prefer the original source name (_lhs_orig_name) over lhs.name,
            # since lhs.name may be an alias created by a prior assignment.
            update_name = _lhs_orig_name or (lhs.name if isinstance(lhs, Value) else None)
            if update_name and update_name in self._variables:
                self._variables[update_name] = rhs
            return rhs
        # Compound assignment: +=, -=, *=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL),
                             (TokKind.SLASH_EQ, BinOp.DIV),
                             (TokKind.PERCENT_EQ, BinOp.MOD),
                             (TokKind.AMP_EQ, BinOp.AND),
                             (TokKind.PIPE_EQ, BinOp.OR),
                             (TokKind.CARET_EQ, BinOp.XOR),
                             (TokKind.LSHIFT_EQ, BinOp.SHL),
                             (TokKind.RSHIFT_EQ, BinOp.SHR)]:
            if self._match(tok_kind):
                rhs = self._parse_assign_expr()
                if isinstance(lhs, Value):
                    new_val = self._new_val(f"{lhs.name}_compound", lhs.ty)
                    self._emit(BinInst(new_val, op, lhs, rhs))
                    update_name = _lhs_orig_name or lhs.name
                    if update_name in self._variables:
                        self._variables[update_name] = new_val
                    return new_val
        return lhs

    def _parse_or_expr(self) -> Operand:
        lhs = self._parse_and_expr()
        while self._match(TokKind.OR):
            rhs = self._parse_and_expr()
            dest = self._new_val("or", INT32)
            self._emit(BinInst(dest, BinOp.OR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_and_expr(self) -> Operand:
        lhs = self._parse_bitor_expr()
        while self._match(TokKind.AND):
            rhs = self._parse_bitor_expr()
            dest = self._new_val("and", INT32)
            self._emit(BinInst(dest, BinOp.AND, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitor_expr(self) -> Operand:
        lhs = self._parse_bitxor_expr()
        while self._match(TokKind.PIPE):
            rhs = self._parse_bitxor_expr()
            folded = self._const_fold(BinOp.OR, lhs, rhs)
            if folded is not None:
                lhs = folded
            else:
                dest = self._new_val("bitor", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.OR, lhs, rhs))
                lhs = dest
        return lhs

    def _parse_bitxor_expr(self) -> Operand:
        lhs = self._parse_bitand_expr()
        while self._match(TokKind.CARET):
            rhs = self._parse_bitand_expr()
            folded = self._const_fold(BinOp.XOR, lhs, rhs)
            if folded is not None:
                lhs = folded
            else:
                dest = self._new_val("bitxor", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.XOR, lhs, rhs))
                lhs = dest
        return lhs

    def _parse_bitand_expr(self) -> Operand:
        lhs = self._parse_cmp_expr()
        while self._match(TokKind.AMP):
            rhs = self._parse_cmp_expr()
            folded = self._const_fold(BinOp.AND, lhs, rhs)
            if folded is not None:
                lhs = folded
            else:
                dest = self._new_val("bitand", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.AND, lhs, rhs))
                lhs = dest
        return lhs

    def _parse_cmp_expr(self) -> Operand:
        lhs = self._parse_shift_expr()
        cmp_ops = {
            TokKind.EQ: CmpOp.EQ, TokKind.NE: CmpOp.NE,
            TokKind.LT: CmpOp.LT, TokKind.LE: CmpOp.LE,
            TokKind.GT: CmpOp.GT, TokKind.GE: CmpOp.GE,
        }
        for tok_kind, cmp_op in cmp_ops.items():
            if self._match(tok_kind):
                rhs = self._parse_add_expr()
                dest = self._new_val("cmp", INT32)
                self._emit(CmpInst(dest, cmp_op, lhs, rhs))
                return dest
        return lhs

    def _result_type(self, a: Operand, b: Operand) -> Type:
        """Determine result type with promotion (wider float wins, float > int)."""
        a_ty = a.ty if isinstance(a, (Value, Const)) else FLOAT
        b_ty = b.ty if isinstance(b, (Value, Const)) else FLOAT
        a_is_float = isinstance(a_ty, ScalarTy) and a_ty.is_float
        b_is_float = isinstance(b_ty, ScalarTy) and b_ty.is_float
        # Float promotion: use the wider float type (double > float > half)
        if a_is_float or b_is_float:
            if a_is_float and b_is_float:
                return a_ty if a_ty.size >= b_ty.size else b_ty
            return a_ty if a_is_float else b_ty
        # Pointer arithmetic
        if isinstance(a_ty, PtrTy):
            return a_ty
        if isinstance(b_ty, PtrTy):
            return b_ty
        # 64-bit promotion
        if (isinstance(a_ty, ScalarTy) and a_ty.size == 8) or (isinstance(b_ty, ScalarTy) and b_ty.size == 8):
            return a_ty if isinstance(a_ty, ScalarTy) and a_ty.size == 8 else b_ty
        # Usual arithmetic conversions: unsigned wins over signed of same width.
        # e.g. INT32 + UINT32 → UINT32 (matches C standard §6.3.1.8).
        if (isinstance(a_ty, ScalarTy) and isinstance(b_ty, ScalarTy)
                and a_ty.size == b_ty.size
                and a_ty.is_signed != b_ty.is_signed):
            return b_ty if a_ty.is_signed else a_ty  # return the unsigned one
        return a_ty

    def _parse_shift_expr(self) -> Operand:
        lhs = self._parse_add_expr()
        while True:
            if self._match(TokKind.LSHIFT):
                rhs = self._parse_add_expr()
                folded = self._const_fold(BinOp.SHL, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    lhs_ty = lhs.ty if isinstance(lhs, (Value, Const)) else INT32
                    dest = self._new_val("shl", lhs_ty)
                    self._emit(BinInst(dest, BinOp.SHL, lhs, rhs))
                    lhs = dest
            elif self._match(TokKind.RSHIFT):
                rhs = self._parse_add_expr()
                folded = self._const_fold(BinOp.SHR, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    lhs_ty = lhs.ty if isinstance(lhs, (Value, Const)) else INT32
                    dest = self._new_val("shr", lhs_ty)
                    self._emit(BinInst(dest, BinOp.SHR, lhs, rhs))
                    lhs = dest
            else:
                break
        return lhs

    def _parse_add_expr(self) -> Operand:
        lhs = self._parse_mul_expr()
        while True:
            if self._match(TokKind.PLUS):
                rhs = self._parse_mul_expr()
                folded = self._const_fold(BinOp.ADD, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("add", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.ADD, lhs, rhs))
                    lhs = dest
            elif self._match(TokKind.MINUS):
                rhs = self._parse_mul_expr()
                folded = self._const_fold(BinOp.SUB, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("sub", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.SUB, lhs, rhs))
                    lhs = dest
            else:
                break
        return lhs

    def _parse_mul_expr(self) -> Operand:
        lhs = self._parse_unary_expr()
        while True:
            if self._match(TokKind.STAR):
                rhs = self._parse_unary_expr()
                folded = self._const_fold(BinOp.MUL, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("mul", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.MUL, lhs, rhs))
                    lhs = dest
            elif self._match(TokKind.SLASH):
                rhs = self._parse_unary_expr()
                folded = self._const_fold(BinOp.DIV, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("div", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.DIV, lhs, rhs))
                    lhs = dest
            elif self._match(TokKind.PERCENT):
                rhs = self._parse_unary_expr()
                folded = self._const_fold(BinOp.MOD, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("mod", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.MOD, lhs, rhs))
                    lhs = dest
            else:
                break
        return lhs

    def _parse_unary_expr(self) -> Operand:
        if self._match(TokKind.AMP):
            # Address-of: &expr — for ptr[idx], return the address not the loaded value
            # Peek: if it's ident[expr], parse as lvalue to get the address
            if self._at(TokKind.IDENT):
                name = self._peek().value
                # &global_sym → materialize the symbol's address into a register
                if name in self._global_consts:
                    cv = self._global_consts[name]
                    if isinstance(cv, SymbolRef):
                        self._advance()  # consume ident
                        addr_val = self._new_val(f"{cv.sym_name}_ptr", cv.ty)
                        self._emit(GlobalAddrInst(addr_val, cv.sym_name, cv.ty.addr_space))
                        return addr_val
                if name in self._variables:
                    var = self._variables[name]
                    if isinstance(var.ty, PtrTy):
                        self._advance()  # consume ident
                        if self._match(TokKind.LBRACKET):
                            index = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            elem_size = var.ty.pointee.size
                            if elem_size != 1:
                                idx_ty = index.ty if isinstance(index, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, index,
                                                   Const(idx_ty, elem_size)))
                                index = scaled
                            addr = self._new_val("addr", var.ty)
                            self._emit(BinInst(addr, BinOp.ADD, var, index))
                            return addr  # return address, no load
                        if self._at(TokKind.ARROW) or self._at(TokKind.DOT):
                            # &p->field or &p.field: postfix operators follow.
                            # Unconsume ident and fall to generic handler so that
                            # _parse_postfix_expr can process ->/. before we spill.
                            self._pos -= 1
                            # fall through to generic handler below
                        else:
                            # No subscript, no member access: &ptr_var → pointer itself
                            return var
                    elif isinstance(var.ty, ScalarTy):
                        # &scalar_local — spill to .local, return pointer to it
                        self._advance()  # consume ident
                        spill_name = f"_spill_{name}"
                        local_ty = PtrTy(var.ty, AddrSpace.LOCAL)
                        if spill_name not in self._variables:
                            spill_val = self._new_val(spill_name, local_ty)
                            self._variables[spill_name] = spill_val
                            if not hasattr(self._kernel, '_local_decls'):
                                self._kernel._local_decls = []
                            self._kernel._local_decls.append((spill_name, var.ty, 1, spill_val))
                        else:
                            spill_val = self._variables[spill_name]
                        # Store current value
                        self._emit(StoreInst(addr=spill_val, value=var))
                        return spill_val
            # Generic fallback: &expr where expr is a scalar (e.g. &b.min_x for struct field)
            operand = self._parse_unary_expr()
            if isinstance(operand, Value) and isinstance(operand.ty, PtrTy):
                return operand
            if (isinstance(operand, Value) and isinstance(operand.ty, ScalarTy)
                    and self._kernel is not None):
                # Spill the scalar to .local memory and return a pointer to it
                spill_name = f"_spill_{operand.name}"
                local_ty = PtrTy(operand.ty, AddrSpace.LOCAL)
                if spill_name not in self._variables:
                    spill_val = self._new_val(spill_name, local_ty)
                    self._variables[spill_name] = spill_val
                    if not hasattr(self._kernel, '_local_decls'):
                        self._kernel._local_decls = []
                    self._kernel._local_decls.append((spill_name, operand.ty, 1, spill_val))
                else:
                    spill_val = self._variables[spill_name]
                self._emit(StoreInst(addr=spill_val, value=operand))
                return spill_val
            return operand
        if self._match(TokKind.STAR):
            operand = self._parse_unary_expr()
            if isinstance(operand, Value) and isinstance(operand.ty, PtrTy):
                dest = self._new_val("deref", operand.ty.pointee)
                self._emit(LoadInst(dest, operand))
                return dest
        if self._match(TokKind.KW_SIZEOF):
            # sizeof(type) or sizeof(expr) — return size as INT32 constant.
            # Always parens: sizeof(T) or sizeof(expr).
            self._expect(TokKind.LPAREN)
            # Try to parse a type first; if that fails, parse an expression.
            saved = self._pos
            size = None
            try:
                ty = self._parse_type_with_ptr()
                size = ty.size if hasattr(ty, 'size') else 4
            except Exception:
                self._pos = saved
                expr = self._parse_expr()
                expr_ty = expr.ty if isinstance(expr, (Value, Const)) else INT32
                size = expr_ty.size if hasattr(expr_ty, 'size') else 4
            self._expect(TokKind.RPAREN)
            return Const(INT32, size)
        if self._match(TokKind.MINUS):
            operand = self._parse_unary_expr()
            # Include Const types: -1LL → dest should be INT64, not INT32
            ty = operand.ty if isinstance(operand, (Value, Const)) else INT32
            dest = self._new_val("neg", ty)
            zero = Const(FLOAT, 0.0) if (isinstance(ty, ScalarTy) and ty.is_float) else Const(ty, 0)
            self._emit(BinInst(dest, BinOp.SUB, zero, operand))
            return dest
        if self._match(TokKind.TILDE):
            operand = self._parse_unary_expr()
            ty = operand.ty if isinstance(operand, (Value, Const)) else INT32
            dest = self._new_val("bnot", ty)
            # XOR with all-ones of the same width for correct NOT semantics
            all_ones = Const(ty, -1) if isinstance(ty, ScalarTy) else Const(INT32, -1)
            self._emit(BinInst(dest, BinOp.XOR, operand, all_ones))
            return dest
        if self._match(TokKind.BANG):
            operand = self._parse_unary_expr()
            dest = self._new_val("lnot", INT32)
            self._emit(CmpInst(dest, CmpOp.EQ, operand, Const(INT32, 0)))
            return dest
        # Prefix ++i / --i — increment/decrement before use
        if self._match(TokKind.PLUSPLUS):
            operand = self._parse_unary_expr()
            if isinstance(operand, Value):
                new_val = self._new_val(f"{operand.name}_preinc", operand.ty)
                self._emit(BinInst(new_val, BinOp.ADD, operand, Const(operand.ty, 1)))
                self._variables[operand.name] = new_val
                return new_val  # pre-increment returns the new value
            return operand
        if self._match(TokKind.MINUSMINUS):
            operand = self._parse_unary_expr()
            if isinstance(operand, Value):
                new_val = self._new_val(f"{operand.name}_predec", operand.ty)
                self._emit(BinInst(new_val, BinOp.SUB, operand, Const(operand.ty, 1)))
                self._variables[operand.name] = new_val
                return new_val  # pre-decrement returns the new value
            return operand
        # Cast: (type)expr
        if self._at(TokKind.LPAREN):
            saved = self._pos
            self._advance()
            try:
                cast_ty = self._parse_type()
                # Check for pointer
                while self._match(TokKind.STAR):
                    cast_ty = PtrTy(cast_ty, AddrSpace.GLOBAL)
                self._expect(TokKind.RPAREN)
                operand = self._parse_unary_expr()
                # Pointer cast: inherit address space from source so
                # (unsigned int *)&local_var emits ld.local, not ld.global
                if (isinstance(cast_ty, PtrTy)
                        and isinstance(operand, Value)
                        and isinstance(operand.ty, PtrTy)
                        and operand.ty.addr_space != AddrSpace.GLOBAL):
                    cast_ty = PtrTy(cast_ty.pointee, operand.ty.addr_space)
                dest = self._new_val("cast", cast_ty)
                self._emit(CvtInst(dest, operand))
                return dest
            except ParseError:
                self._pos = saved
        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Operand:
        lhs = self._parse_primary_expr()

        while True:
            # i++ / i--
            if self._match(TokKind.PLUSPLUS):
                if isinstance(lhs, Value):
                    old = lhs
                    new_val = self._new_val(f"{old.name}_inc", old.ty)
                    self._emit(BinInst(new_val, BinOp.ADD, old, Const(old.ty, 1)))
                    self._variables[old.name] = new_val
                    lhs = old  # post-increment returns old value
                continue
            if self._match(TokKind.MINUSMINUS):
                if isinstance(lhs, Value):
                    old = lhs
                    new_val = self._new_val(f"{old.name}_dec", old.ty)
                    self._emit(BinInst(new_val, BinOp.SUB, old, Const(old.ty, 1)))
                    self._variables[old.name] = new_val
                    lhs = old
                continue

            if self._match(TokKind.LBRACKET):
                # Array indexing: ptr[index]
                index = self._parse_expr()
                self._expect(TokKind.RBRACKET)
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                    # Multi-dim array check: if this var has a row stride AND a second
                    # '[' follows, use row stride (not elem_size) and keep as pointer.
                    row_stride = self._array_row_strides.get(lhs.name)
                    if row_stride is not None and self._at(TokKind.LBRACKET):
                        idx_ty = index.ty if isinstance(index, Value) else INT32
                        scaled = self._new_val("scale", idx_ty)
                        self._emit(BinInst(scaled, BinOp.MUL, index, Const(idx_ty, row_stride)))
                        addr = self._new_val("addr", lhs.ty)
                        self._emit(BinInst(addr, BinOp.ADD, lhs, scaled))
                        lhs = addr  # keep as pointer for chained [j] to follow
                        continue
                    elem_size = lhs.ty.pointee.size
                    # addr = base + index * elem_size
                    if elem_size != 1:
                        idx_ty = index.ty if isinstance(index, Value) else INT32
                        scaled = self._new_val("scale", idx_ty)
                        self._emit(BinInst(scaled, BinOp.MUL, index, Const(idx_ty, elem_size)))
                        index = scaled
                    addr = self._new_val("addr", lhs.ty)
                    self._emit(BinInst(addr, BinOp.ADD, lhs, index))
                    if isinstance(lhs.ty.pointee, StructTy):
                        # Struct element: keep as pointer so .field access can
                        # compute the correct byte offset and load the scalar.
                        lhs = addr
                    else:
                        # Load the scalar value
                        dest = self._new_val("elem", lhs.ty.pointee)
                        self._emit(LoadInst(dest, addr))
                        lhs = dest
            elif self._match(TokKind.DOT):
                member = self._expect(TokKind.IDENT).value
                # Built-in: threadIdx.x, blockIdx.y, etc.
                if isinstance(lhs, Value) and lhs.name in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim'):
                    builtin = f"{lhs.name}.{member}"
                    # CUDA hardware registers are unsigned 32-bit; using UINT32 ensures
                    # comparisons emit setp.lt.u32 (correct) not setp.lt.s32.
                    dest = self._new_val(builtin.replace('.', '_'), UINT32)
                    self._emit(CallInst(dest, builtin))
                    lhs = dest
                # Struct / vector variable member access (e.g. v.x where v is float3).
                # Each field is a separate scalar variable: _variables['v_x'], etc.
                elif isinstance(lhs, Value) and isinstance(lhs.ty, StructTy):
                    sty = lhs.ty
                    var_name = lhs.name
                    # Check for inline array member: float data[N] expanded as data_0..data_{N-1}
                    arr_info = self._struct_field_arrays.get(sty.name, {})
                    if member in arr_info and self._at(TokKind.LBRACKET):
                        # v.data[i] where data is an array member
                        self._advance()  # consume '['
                        idx_expr = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        elem_ty = sty.field_type(f"{member}_0")
                        n = arr_info[member]
                        if isinstance(idx_expr, Const):
                            k = int(idx_expr.value) % n
                            key = f"{var_name}_{member}_{k}"
                            if key not in self._variables:
                                fval = self._new_val(key, elem_ty)
                                self._variables[key] = fval
                            lhs = self._variables[key]
                        else:
                            # Dynamic index into struct array member: return field_0 as fallback
                            key0 = f"{var_name}_{member}_0"
                            if key0 not in self._variables:
                                fval = self._new_val(key0, elem_ty)
                                self._variables[key0] = fval
                            lhs = self._variables[key0]
                    else:
                        field_ty = sty.field_type(member)
                        field_key = f"{lhs.name}_{member}"
                        if field_key in self._variables:
                            lhs = self._variables[field_key]
                        else:
                            # Field not yet created — create it lazily
                            dest = self._new_val(field_key, field_ty)
                            self._variables[field_key] = dest
                            lhs = dest
                elif isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy) and isinstance(lhs.ty.pointee, StructTy):
                    sty = lhs.ty.pointee
                    arr_info = self._struct_field_arrays.get(sty.name, {})
                    if member in arr_info and self._at(TokKind.LBRACKET):
                        # ptr.data[i] where data is an array member
                        self._advance()  # consume '['
                        idx_expr = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        elem_ty = sty.field_type(f"{member}_0")
                        n = arr_info[member]
                        if isinstance(idx_expr, Const):
                            k = int(idx_expr.value) % n
                            field_off = sty.field_offset(f"{member}_{k}")
                        else:
                            field_off = sty.field_offset(f"{member}_0")
                        addr = self._new_val("faddr", PtrTy(elem_ty, lhs.ty.addr_space))
                        self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        dest = self._new_val(f"{member}", elem_ty)
                        self._emit(LoadInst(dest, addr))
                        lhs = dest
                    else:
                        field_off = sty.field_offset(member)
                        field_ty = sty.field_type(member)
                        # ptr.field: compute address of field
                        addr = self._new_val("faddr", PtrTy(field_ty, lhs.ty.addr_space))
                        self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        if isinstance(field_ty, StructTy):
                            # Nested struct: return the pointer (so next .field access works)
                            lhs = addr
                        else:
                            # Scalar field: load the value
                            dest = self._new_val(f"{member}", field_ty)
                            self._emit(LoadInst(dest, addr))
                            lhs = dest
            elif self._match(TokKind.ARROW):
                # ptr->field: sugar for (*ptr).field — lhs must be a pointer to struct.
                member = self._expect(TokKind.IDENT).value
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy) and isinstance(lhs.ty.pointee, StructTy):
                    sty = lhs.ty.pointee
                    arr_info = self._struct_field_arrays.get(sty.name, {})
                    if member in arr_info and self._at(TokKind.LBRACKET):
                        # ptr->data[i] where data is an array member
                        self._advance()  # consume '['
                        idx_expr = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        elem_ty = sty.field_type(f"{member}_0")
                        n = arr_info[member]
                        if isinstance(idx_expr, Const):
                            k = int(idx_expr.value) % n
                            field_off = sty.field_offset(f"{member}_{k}")
                        else:
                            field_off = sty.field_offset(f"{member}_0")
                        addr = self._new_val("faddr", PtrTy(elem_ty, lhs.ty.addr_space))
                        self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        dest = self._new_val(f"{member}", elem_ty)
                        self._emit(LoadInst(dest, addr))
                        lhs = dest
                    else:
                        field_off = sty.field_offset(member)
                        field_ty = sty.field_type(member)
                        addr = self._new_val("faddr", PtrTy(field_ty, lhs.ty.addr_space))
                        self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        if isinstance(field_ty, StructTy):
                            lhs = addr
                        else:
                            dest = self._new_val(f"{member}", field_ty)
                            self._emit(LoadInst(dest, addr))
                            lhs = dest
                # else: non-struct pointer arrow — treat as unknown, leave lhs unchanged
            else:
                break
        return lhs

    def _parse_primary_expr(self) -> Operand:
        tok = self._peek()

        if tok.kind == TokKind.CHAR_LIT:
            self._advance()
            raw = tok.value[1:-1]  # strip quotes
            _ESCAPE = {'n': 10, 't': 9, 'r': 13, '0': 0, '\\': 92, "'": 39, '"': 34}
            if len(raw) >= 2 and raw[0] == '\\':
                ival = _ESCAPE.get(raw[1], ord(raw[1]))
            else:
                ival = ord(raw[0]) if raw else 0
            return Const(INT32, ival)

        if tok.kind == TokKind.KW_TRUE:
            self._advance()
            return Const(INT32, 1)

        if tok.kind == TokKind.KW_FALSE:
            self._advance()
            return Const(INT32, 0)

        if tok.kind == TokKind.INT_LIT:
            self._advance()
            raw = tok.value
            # Detect suffixes: u/U (unsigned), l/L or ll/LL (long/long long)
            suffix = ''
            for ch in reversed(raw.lower()):
                if ch in ('u', 'l'):
                    suffix = ch + suffix
                else:
                    break
            has_unsigned_suffix = 'u' in suffix
            has_ll_suffix = 'll' in suffix or 'l' in suffix
            stripped = raw.rstrip('uUlL')
            # Python 3 requires 0o prefix for octal; C uses 0 prefix.
            # Convert C-style octal 0755 → 0o755 before parsing.
            if (len(stripped) > 1 and stripped[0] == '0'
                    and stripped[1] not in ('x', 'X', 'b', 'B', 'o', 'O')):
                stripped = '0o' + stripped[1:]
            val = int(stripped, 0)
            # C standard §6.4.4.1 type selection:
            # - ll/LL suffix OR value > UINT32_MAX → 64-bit type
            # - u/U suffix  → unsigned (UINT32 or UINT64)
            # - Otherwise:  signed (INT32 if ≤ INT32_MAX, else UINT32 for hex
            #               that exceeds INT32_MAX, or INT64 for larger values)
            if has_ll_suffix or val > 0xFFFFFFFF:
                if has_unsigned_suffix:
                    return Const(UINT64, val)
                else:
                    return Const(INT64, val)
            if has_unsigned_suffix or val > 0x7FFFFFFF:
                return Const(UINT32, val)
            return Const(INT32, val)

        if tok.kind == TokKind.FLOAT_LIT:
            self._advance()
            val = float(tok.value.rstrip('fF'))
            return Const(FLOAT, val)

        if tok.kind == TokKind.IDENT:
            name = tok.value
            self._advance()

            # C++ namespace::name scope resolution — strip namespace qualifier,
            # use only the unqualified name that follows.
            while (self._at(TokKind.COLON)
                   and self._pos + 1 < len(self._toks)
                   and self._toks[self._pos + 1].kind == TokKind.COLON):
                self._advance()  # consume first ':'
                self._advance()  # consume second ':'
                if self._at(TokKind.IDENT):
                    name = self._peek().value
                    self._advance()

            # C++ nullptr → null pointer constant (UINT64 0)
            if name == 'nullptr':
                return Const(UINT64, 0)

            # C++ static_cast<Type>(expr) → cast expression
            if name in ('static_cast', 'reinterpret_cast', 'const_cast') and self._at(TokKind.LT):
                self._advance()  # consume '<'
                try:
                    cast_ty = self._parse_type()
                    while self._match(TokKind.STAR):
                        cast_ty = PtrTy(cast_ty, AddrSpace.GLOBAL)
                except Exception:
                    cast_ty = INT32
                self._expect(TokKind.GT)
                self._expect(TokKind.LPAREN)
                operand = self._parse_assign_expr()
                self._expect(TokKind.RPAREN)
                # Pointer cast: inherit address space from source
                if (isinstance(cast_ty, PtrTy)
                        and isinstance(operand, Value)
                        and isinstance(operand.ty, PtrTy)
                        and operand.ty.addr_space != AddrSpace.GLOBAL):
                    cast_ty = PtrTy(cast_ty.pointee, operand.ty.addr_space)
                dest = self._new_val("cast", cast_ty)
                self._emit(CvtInst(dest, operand))
                return dest

            # Check for function call
            if self._match(TokKind.LPAREN):
                args = []
                if not self._at(TokKind.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TokKind.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokKind.RPAREN)

                if name == '__syncthreads':
                    self._emit(CallInst(None, '__syncthreads', args))
                    return Const(VOID, 0)
                elif name in ('__all_sync', '__any_sync'):
                    # __all_sync(mask, pred) → 1 if all lanes have pred set
                    # __any_sync(mask, pred) → 1 if any lane has pred set
                    dest = self._new_val(name.replace('__', ''), INT32)
                    self._emit(CallInst(dest, name, args))
                    return dest
                elif name in ('__shfl_sync', '__shfl_up_sync', '__shfl_down_sync',
                              '__shfl_xor_sync', '__ballot_sync'):
                    # __ballot_sync returns a u32 lane mask (bitmask of active lanes).
                    # __shfl_*_sync shuffles 32-bit values bitwise; the return type must
                    # match the shuffled value's type so that floats stay as floats.
                    # args: (mask, value, offset[, width]) — value is args[1].
                    if name == '__ballot_sync':
                        ret_ty = UINT32
                    else:
                        # Infer from the shuffled value argument (args[1])
                        val_arg = args[1] if len(args) > 1 else None
                        if isinstance(val_arg, (Value, Const)) and isinstance(val_arg.ty, ScalarTy):
                            ret_ty = val_arg.ty
                        else:
                            ret_ty = INT32  # fallback
                    dest = self._new_val(name.replace('__',''), ret_ty)
                    self._emit(CallInst(dest, name, args))
                    return dest
                elif name in ('atomicAdd', 'atomicSub', 'atomicMin', 'atomicMax',
                              'atomicAnd', 'atomicOr', 'atomicXor', 'atomicExch',
                              'atomicCAS'):
                    # Atomic operations: atomicAdd(addr, val) → atom.global.add
                    if len(args) >= 2:
                        addr = args[0]
                        val = args[1]
                        result_ty = val.ty if isinstance(val, Value) else INT32
                        dest = self._new_val(f"atomic_{name}", result_ty)
                        self._emit(CallInst(dest, name, args))
                        return dest
                    return Const(INT32, 0)
                elif name == '__ldg':
                    # __ldg(ptr) — load with non-coherent (read-only) cache
                    if args:
                        ptr_arg = args[0]
                        if isinstance(ptr_arg, Value) and isinstance(ptr_arg.ty, PtrTy):
                            # Re-wrap with CONST addr space for nc load
                            nc_ptr = Value(ptr_arg.name, PtrTy(ptr_arg.ty.pointee, AddrSpace.CONST), ptr_arg.id)
                            dest = self._new_val("ldg", ptr_arg.ty.pointee)
                            self._emit(LoadInst(dest, nc_ptr))
                            return dest
                    return Const(INT32, 0)
                elif name == 'printf':
                    # printf("fmt", args...) — emit PrintfInst
                    # The first arg was STRING_LIT; it set self._last_string_lit
                    fmt_str = getattr(self, '_last_string_lit', '')
                    printf_args = args[1:] if len(args) > 1 else []
                    self._emit(PrintfInst(fmt_str, printf_args))
                    return Const(VOID, 0)
                elif hasattr(self, '_device_funcs') and name in self._device_funcs:
                    # Inline __device__ function with multi-return support
                    dfunc = self._device_funcs[name]
                    saved_vars = dict(self._variables)
                    saved_pos = self._pos
                    saved_inline_target = self._inline_return_target

                    # Bind arguments to parameters
                    for (pname, pty), arg in zip(dfunc['params'], args):
                        self._variables[pname] = arg

                    # Create return destination and merge block
                    ret_ty = dfunc['ret_ty']
                    return_dest = self._new_val(f"{name}_ret", ret_ty)
                    return_merge = self._new_block(f"inline_{name}_merge")
                    self._inline_return_target = (return_dest, return_merge.label)

                    # Parse entire body using normal statement parsing
                    self._pos = dfunc['body_start']
                    self._expect(TokKind.LBRACE)
                    while not self._at(TokKind.RBRACE):
                        self._parse_stmt()
                    # Consume closing brace
                    self._advance()

                    # If current block has no terminator, branch to merge
                    if self._cur_block.terminator is None:
                        self._cur_block.terminator = BrTerm(return_merge.label)

                    self._cur_block = return_merge
                    self._pos = saved_pos
                    self._variables = saved_vars
                    self._inline_return_target = saved_inline_target
                    return return_dest
                else:
                    # Infer return type for known math intrinsics; fallback INT32.
                    _void_stmts   = ('__threadfence', '__threadfence_block',
                                     '__threadfence_system')
                    _float_unary  = ('sqrtf','rsqrtf','rcpf','fabsf','sinf','cosf',
                                     'tanf','expf','exp2f','exp10f','logf','log2f','log10f',
                                     'floorf','ceilf','roundf','truncf',
                                     'sqrt','rsqrt','fabs','sin','cos',
                                     'exp','exp2','exp10','log','log2','log10',
                                     'floor','ceil','round','trunc')
                    _float_binary = ('fminf','fmaxf','fmodf','powf',
                                     'fmin','fmax','fmod','pow','hypotf','atan2f')
                    _float_ternary = ('fmaf', 'fma')
                    _int_unary    = ('abs',)
                    _int_binary   = ('min','max')
                    _uint_return  = ('__activemask',)
                    _sync_ops     = ('__syncwarp',)
                    _int_ops      = ('__popc', '__popcll', '__clz', '__clzll',
                                     '__brev', '__brevll', '__ffs', '__ffsll')
                    if name in _void_stmts:
                        self._emit(CallInst(None, name, args))
                        return Const(VOID, 0)
                    elif name in _sync_ops:
                        self._emit(CallInst(None, name, args))
                        return Const(VOID, 0)
                    elif name in _uint_return:
                        dest = self._new_val(name, UINT32)
                        self._emit(CallInst(dest, name, args))
                        return dest
                    elif name in _int_ops:
                        # __popc/__clz: int result, arg width determines PTX type
                        # __brev:   unsigned int result (same width as arg)
                        # __brevll: unsigned long long result (64-bit)
                        # __popcll/__clzll: int result, but arg is 64-bit
                        if name == '__brevll':
                            ret_ty = UINT64
                        elif name in ('__popcll', '__clzll'):
                            ret_ty = INT32
                        elif name == '__brev':
                            ret_ty = UINT32
                        else:
                            ret_ty = INT32
                        dest = self._new_val(name, ret_ty)
                        self._emit(CallInst(dest, name, args))
                        return dest
                    elif name in _float_unary:
                        ret_ty = FLOAT
                    elif name in _float_binary:
                        ret_ty = FLOAT
                    elif name in _float_ternary:
                        ret_ty = FLOAT
                    elif name in _int_unary:
                        a_ty = args[0].ty if args and isinstance(args[0], (Value, Const)) else INT32
                        ret_ty = a_ty
                    elif name in _int_binary:
                        a_ty = args[0].ty if args and isinstance(args[0], (Value, Const)) else INT32
                        ret_ty = self._result_type(args[0], args[1]) if args and len(args) > 1 else a_ty
                    else:
                        ret_ty = INT32
                    dest = self._new_val(name, ret_ty)
                    self._emit(CallInst(dest, name, args))
                    return dest

            # Lazy param loading: emit ld.param on first use
            if hasattr(self, '_lazy_params') and name in self._lazy_params and name not in self._variables:
                idx, p = self._lazy_params[name]
                val = self._new_val(p.name, p.ty)
                self._emit(ParamInst(val, idx, p.name))
                self._variables[p.name] = val

            # Variable reference
            if name in self._variables:
                return self._variables[name]

            # Module-level compile-time constants (enum values, etc.)
            if name in self._global_consts:
                cv = self._global_consts[name]
                if isinstance(cv, SymbolRef):
                    # Device/constant global: materialize to a register.
                    # If followed by '[', the caller will do array indexing so
                    # we need a PtrTy Value (emit GlobalAddrInst).
                    # Otherwise auto-load scalar values.
                    if self._at(TokKind.LBRACKET):
                        addr_val = self._new_val(f"{cv.sym_name}_ptr", cv.ty)
                        self._emit(GlobalAddrInst(addr_val, cv.sym_name, cv.ty.addr_space))
                        return addr_val
                    elif isinstance(cv.ty.pointee, ScalarTy):
                        # Scalar global used as rvalue: emit load directly.
                        dest = self._new_val(cv.sym_name, cv.ty.pointee)
                        self._emit(LoadInst(dest, cv))
                        return dest
                    else:
                        addr_val = self._new_val(f"{cv.sym_name}_ptr", cv.ty)
                        self._emit(GlobalAddrInst(addr_val, cv.sym_name, cv.ty.addr_space))
                        return addr_val
                return cv

            # Built-in names (threadIdx, blockIdx, blockDim)
            if name in ('threadIdx', 'blockIdx', 'blockDim', 'gridDim'):
                return Value(name, INT32)  # placeholder, resolved by .member access

            raise ParseError(f"Line {tok.line}: undefined variable '{name}'")

        if tok.kind == TokKind.STRING_LIT:
            self._advance()
            # Strip surrounding quotes and process escape sequences
            raw = tok.value[1:-1]  # remove leading/trailing "
            processed = raw.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\\\', '\\').replace('\\"', '"')
            self._last_string_lit = processed
            return Const(VOID, 0)  # placeholder

        if tok.kind == TokKind.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokKind.RPAREN)
            return expr

        raise ParseError(f"Line {tok.line}: unexpected token '{tok.value}'")

    # -- Statement parsing ---------------------------------------------------

    def _parse_stmt(self):
        tok = self._peek()

        # const/volatile/static/inline/register declaration: skip the qualifier and parse as normal
        _ignorable_quals = (TokKind.KW_CONST, TokKind.KW_VOLATILE, TokKind.KW_STATIC)
        while tok.kind in _ignorable_quals:
            self._advance()
            tok = self._peek()  # re-read after consuming qualifier(s)

        # Empty statement: ; (null statement, e.g. from macro expanding to nothing)
        if tok.kind == TokKind.SEMI:
            self._advance()
            return

        # goto label; — skip (CFG goto not supported, treat as no-op)
        if tok.kind == TokKind.KW_GOTO:
            self._advance()  # consume 'goto'
            if self._at(TokKind.IDENT):
                self._advance()  # consume label name
            self._match(TokKind.SEMI)
            return

        # Label declaration: identifier: (followed by anything)
        # Detect IDENT COLON that is not a switch case/default
        if (tok.kind == TokKind.IDENT
                and self._pos + 1 < len(self._toks)
                and self._toks[self._pos + 1].kind == TokKind.COLON):
            self._advance()  # consume label name
            self._advance()  # consume ':'
            return  # label is a no-op in our flat SSA IR

        # Bare block: { stmt; stmt; ... }  — anonymous compound statement
        if tok.kind == TokKind.LBRACE:
            self._parse_stmt_or_block()
            return

        # Inline assembly: asm(...); or __asm__(...); — skip entirely
        if tok.kind == TokKind.IDENT and tok.value in ('asm', '__asm__', '__asm'):
            # Consume tokens until the matching ';', balancing parentheses
            self._advance()  # consume 'asm'
            depth = 0
            while not self._at(TokKind.EOF):
                k = self._peek().kind
                if k == TokKind.LPAREN:
                    depth += 1; self._advance()
                elif k == TokKind.RPAREN:
                    depth -= 1; self._advance()
                    if depth == 0:
                        break
                elif k == TokKind.SEMI and depth == 0:
                    break
                else:
                    self._advance()
            self._match(TokKind.SEMI)
            return

        # __shared__ declaration: __shared__ type name[size], name[d0][d1]...,
        # or extern __shared__ type name[] (dynamic shared memory, size=0 sentinel).
        if tok.kind == TokKind.KW_SHARED:
            self._advance()
            ty = self._parse_type()
            name = self._expect(TokKind.IDENT).value
            self._expect(TokKind.LBRACKET)
            # extern __shared__ float sdata[]; — empty brackets → dynamic
            if self._at(TokKind.RBRACKET):
                self._advance()
                self._expect(TokKind.SEMI)
                smem_ty = PtrTy(ScalarTy(ScalarType.FLOAT) if ty == FLOAT else ty, AddrSpace.SHARED)
                val = self._new_val(name, smem_ty)
                self._variables[name] = val
                if not hasattr(self._kernel, '_shared_decls'):
                    self._kernel._shared_decls = []
                self._kernel._shared_decls.append((name, ty, 0))  # size=0 → extern/dynamic
                return
            size_op = self._parse_assign_expr()
            d0 = int(size_op.value) if isinstance(size_op, Const) else 1
            size = d0
            self._expect(TokKind.RBRACKET)
            # Handle multi-dimensional arrays [d0][d1]... — collapse all dims
            # into one flat array of total_elements elements, tracking inner product
            # for multi-dim index stride computation.
            inner_dims = []
            while self._at(TokKind.LBRACKET):
                self._advance()
                dim_op = self._parse_assign_expr()
                dim = int(dim_op.value) if isinstance(dim_op, Const) else 1
                inner_dims.append(dim)
                size *= dim
                self._expect(TokKind.RBRACKET)
            # If multi-dim: row stride = product(inner_dims) * elem_size
            if inner_dims:
                elem_size = ty.size if isinstance(ty, ScalarTy) else 8
                inner_prod = 1
                for d in inner_dims:
                    inner_prod *= d
                self._array_row_strides[name] = inner_prod * elem_size
            self._expect(TokKind.SEMI)
            # Create a shared-memory pointer variable
            smem_ty = PtrTy(ScalarTy(ScalarType.FLOAT) if ty == FLOAT else ty, AddrSpace.SHARED)
            val = self._new_val(name, smem_ty)
            self._variables[name] = val
            # Store smem info for codegen (size in bytes)
            if not hasattr(self._kernel, '_shared_decls'):
                self._kernel._shared_decls = []
            self._kernel._shared_decls.append((name, ty, size))
            return

        # Variable declaration: type name [= expr] [, name2 [= expr2]] ...;
        # Handles both single and multiple comma-separated declarators.
        # Also handles typedef'd types (e.g. float3, int2, user typedefs).
        if (tok.kind in (TokKind.KW_INT, TokKind.KW_UNSIGNED, TokKind.KW_SIGNED,
                         TokKind.KW_FLOAT, TokKind.KW_DOUBLE, TokKind.KW_VOID,
                         TokKind.KW_LONG, TokKind.KW_HALF, TokKind.KW_CHAR,
                         TokKind.KW_SHORT, TokKind.KW_BOOL,
                         TokKind.KW_STRUCT, TokKind.KW_UNION)
                or (tok.kind == TokKind.IDENT and (tok.value in self._typedefs
                                                    or tok.value in self._struct_types))):
            ty = self._parse_type_with_ptr()
            while True:
                # Each declarator may have its own pointer stars: int *a, b, *c;
                decl_ty = ty
                while self._match(TokKind.STAR):
                    decl_ty = PtrTy(decl_ty, AddrSpace.GLOBAL)
                name = self._expect(TokKind.IDENT).value

                # Struct / vector type variable (e.g. float3 v;).
                # Decompose into per-field scalar variables: v_x, v_y, v_z.
                # A sentinel Value with StructTy is kept for type resolution
                # during member-access parsing (.x / .y / .z / .w).
                if isinstance(decl_ty, StructTy):
                    # Struct array: Foo arr[N] — allocate in .local, expose as PtrTy(Foo, LOCAL)
                    if self._at(TokKind.LBRACKET):
                        self._advance()
                        sz_op = self._parse_assign_expr()
                        count = int(sz_op.value) if isinstance(sz_op, Const) else 1
                        self._expect(TokKind.RBRACKET)
                        arr_ty = PtrTy(decl_ty, AddrSpace.LOCAL)
                        arr_val = self._new_val(name, arr_ty)
                        self._variables[name] = arr_val
                        if not hasattr(self._kernel, '_local_decls'):
                            self._kernel._local_decls = []
                        self._kernel._local_decls.append((name, decl_ty, count, arr_val))
                        if not self._match(TokKind.COMMA):
                            break
                        continue
                    sentinel = self._new_val(name, decl_ty)
                    self._variables[name] = sentinel
                    for fname, fty in decl_ty.fields:
                        fval = self._new_val(f"{name}_{fname}", fty)
                        self._variables[f"{name}_{fname}"] = fval
                    # Handle initializer: Vec3 v = {1.0f, 2.0f, 3.0f}; or v = src_ptr[i];
                    if self._match(TokKind.ASSIGN):
                        if self._at(TokKind.LBRACE):
                            # Aggregate initializer: { expr, expr, ... }
                            self._advance()  # consume '{'
                            scalar_fields = [(fname, fty) for fname, fty in decl_ty.fields
                                             if isinstance(fty, ScalarTy)]
                            i = 0
                            while not self._at(TokKind.RBRACE) and not self._at(TokKind.EOF):
                                val_expr = self._parse_assign_expr()
                                if i < len(scalar_fields):
                                    fname, fty = scalar_fields[i]
                                    fval = self._new_val(f"{name}_{fname}", fty)
                                    if isinstance(val_expr, Const) and val_expr.ty != fty:
                                        from_const = Const(fty, val_expr.value)
                                        fval2 = self._new_val(f"{name}_{fname}", fty)
                                        self._emit(BinInst(fval2, BinOp.ADD, from_const, Const(fty, 0.0 if fty.is_float else 0)))
                                        self._variables[f"{name}_{fname}"] = fval2
                                    else:
                                        self._emit(BinInst(fval, BinOp.ADD, val_expr,
                                                           Const(fty, 0.0 if fty.is_float else 0)))
                                        self._variables[f"{name}_{fname}"] = fval
                                i += 1
                                if not self._match(TokKind.COMMA):
                                    break
                            self._expect(TokKind.RBRACE)
                        else:
                            rhs = self._parse_assign_expr()
                            # rhs should be a PtrTy(StructTy) address pointing to the source
                            if isinstance(rhs, Value) and isinstance(rhs.ty, PtrTy) and isinstance(rhs.ty.pointee, StructTy):
                                sty = rhs.ty.pointee
                                for fname, fty in sty.fields:
                                    if isinstance(fty, ScalarTy):
                                        off = sty.field_offset(fname)
                                        faddr = self._new_val("faddr", PtrTy(fty, rhs.ty.addr_space))
                                        self._emit(BinInst(faddr, BinOp.ADD, rhs, Const(INT32, off)))
                                        loaded = self._new_val(f"{name}_{fname}", fty)
                                        self._emit(LoadInst(loaded, faddr))
                                        self._variables[f"{name}_{fname}"] = loaded
                    if not self._match(TokKind.COMMA):
                        break
                    continue

                # Local array declaration: type name[N] or name[d0][d1]...;
                # Allocate in .local memory, expose as a pointer.
                if self._at(TokKind.LBRACKET):
                    self._advance()
                    size_operand = self._parse_assign_expr()
                    self._expect(TokKind.RBRACKET)
                    d0 = int(size_operand.value) if isinstance(size_operand, Const) else 1
                    size = d0
                    inner_dims = []
                    while self._at(TokKind.LBRACKET):
                        self._advance()
                        dim_op = self._parse_assign_expr()
                        dim = int(dim_op.value) if isinstance(dim_op, Const) else 1
                        inner_dims.append(dim)
                        size *= dim
                        self._expect(TokKind.RBRACKET)
                    if inner_dims:
                        elem_size = decl_ty.size if isinstance(decl_ty, ScalarTy) else 8
                        inner_prod = 1
                        for d in inner_dims:
                            inner_prod *= d
                        self._array_row_strides[name] = inner_prod * elem_size
                    arr_ty = PtrTy(decl_ty, AddrSpace.LOCAL)
                    val = self._new_val(name, arr_ty)
                    self._variables[name] = val
                    # Store local array info for codegen
                    if not hasattr(self._kernel, '_local_decls'):
                        self._kernel._local_decls = []
                    self._kernel._local_decls.append((name, decl_ty, size, val))
                    # Optional aggregate initializer: int arr[N] = {e0, e1, ...};
                    if self._match(TokKind.ASSIGN):
                        if self._at(TokKind.LBRACE):
                            self._advance()  # consume '{'
                            elem_sz = decl_ty.size if isinstance(decl_ty, ScalarTy) else 8
                            init_idx = 0
                            while not self._at(TokKind.RBRACE) and not self._at(TokKind.EOF):
                                elem_val = self._parse_assign_expr()
                                if init_idx < size:
                                    offset = Const(INT32, init_idx * elem_sz)
                                    elem_addr = self._new_val("earr", arr_ty)
                                    self._emit(BinInst(elem_addr, BinOp.ADD, val, offset))
                                    if isinstance(elem_val, Const) and elem_val.ty != decl_ty:
                                        elem_val = Const(decl_ty, elem_val.value)
                                    self._emit(StoreInst(addr=elem_addr, value=elem_val))
                                init_idx += 1
                                if not self._match(TokKind.COMMA):
                                    break
                            self._expect(TokKind.RBRACE)
                    if not self._match(TokKind.COMMA):
                        break
                    continue

                val = self._new_val(name, decl_ty)
                self._variables[name] = val

                if self._match(TokKind.ASSIGN):
                    rhs = self._parse_assign_expr()
                    # For simplicity: treat the variable as the RHS value directly
                    self._variables[name] = rhs if isinstance(rhs, Value) else val
                    if isinstance(rhs, Const):
                        # Need to materialize the constant
                        self._emit(BinInst(val, BinOp.ADD, rhs, Const(decl_ty, 0)))
                        self._variables[name] = val
                    elif isinstance(rhs, Value) and rhs != val:
                        # Insert a CvtInst when the declared type differs from rhs:
                        #   - same-width signedness mismatch:  int x = uint_expr
                        #   - widening:                        long long v = int_expr
                        #   - int→float:                       float f = int_expr
                        # In all other cases, use rhs directly (no copy).
                        rhs_ty = rhs.ty
                        need_cvt = False
                        if isinstance(decl_ty, ScalarTy) and isinstance(rhs_ty, ScalarTy):
                            if decl_ty != rhs_ty:
                                # Any scalar type mismatch: coerce to declared type
                                need_cvt = True
                        if need_cvt:
                            self._emit(CvtInst(val, rhs))
                            self._variables[name] = val
                        else:
                            self._variables[name] = rhs

                if not self._match(TokKind.COMMA):
                    break
            self._expect(TokKind.SEMI)
            return

        # If statement
        if tok.kind == TokKind.KW_IF:
            self._advance()
            self._expect(TokKind.LPAREN)
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)

            true_bb = self._new_block("if_true")
            false_bb = self._new_block("if_false")
            merge_bb = self._new_block("if_merge")

            self._cur_block.terminator = CondBrTerm(cond, true_bb.label, false_bb.label)

            # Snapshot variables before branches so that:
            # (a) each branch starts from the same state,
            # (b) values modified in one branch are written back to their
            #     canonical pre-if register before branching to the merge, and
            # (c) the merge block always sees canonical (pre-if) bindings.
            vars_before_if = dict(self._variables)

            self._cur_block = true_bb
            self._parse_stmt_or_block()
            # Always writeback modified variables to their canonical pre-if Values,
            # even when the branch already has a terminator (break/continue/return).
            # This ensures that modifications before a break/continue are persisted
            # into the canonical registers that the outer loop's writeback relies on.
            self._loop_writeback(vars_before_if)
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(merge_bb.label)

            # Reset to pre-if state so the false branch starts from the same
            # variable environment as the true branch did.
            self._variables = dict(vars_before_if)

            self._cur_block = false_bb
            if self._match(TokKind.KW_ELSE):
                self._parse_stmt_or_block()
            self._loop_writeback(vars_before_if)
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(merge_bb.label)

            # Restore canonical bindings at the merge block.
            self._variables = dict(vars_before_if)
            self._cur_block = merge_bb
            return

        # For loop — uses mutable variable model
        # Variables modified in the loop body/increment are written back to their
        # canonical Value so the condition block always reads the current value.
        if tok.kind == TokKind.KW_FOR:
            self._advance()
            self._expect(TokKind.LPAREN)

            # Snapshot variables before init to track which get modified
            vars_before = dict(self._variables)

            # Parse init statement
            self._parse_stmt()

            # Save token positions for condition and increment
            cond_start = self._pos
            depth = 0
            while not (self._peek().kind == TokKind.SEMI and depth == 0):
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            self._advance()  # skip ;

            inc_start = self._pos
            depth = 0
            while not (self._peek().kind == TokKind.RPAREN and depth == 0):
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            self._expect(TokKind.RPAREN)
            body_resume = self._pos

            # Snapshot variables after init (these are the "loop variables")
            loop_vars = dict(self._variables)

            # Build CFG
            cond_bb = self._new_block("for_cond")
            body_bb = self._new_block("for_body")
            inc_bb = self._new_block("for_inc")
            exit_bb = self._new_block("for_exit")

            self._cur_block.terminator = BrTerm(cond_bb.label)

            # Emit condition — must read current loop variable values
            self._cur_block = cond_bb
            self._pos = cond_start
            # Empty condition: for(;;) — always true, loop exits only via break
            if self._toks[cond_start].kind == TokKind.SEMI:
                cond = Const(INT32, 1)
            else:
                cond = self._parse_expr()
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            # Emit body (with break → exit_bb, continue → inc_bb)
            self._pos = body_resume
            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(None)
            self._continue_targets.append(inc_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()
            if self._cur_block.terminator is None:
                # Natural fall-through from the body's last block to inc_bb.
                # Emit writebacks for body-modified variables HERE, while still
                # in that block, so the written-back values dominate their uses
                # in inc_bb and cond_bb.  (If the body exits via break/continue
                # the if-branch writeback mechanism already persisted those values.)
                self._loop_writeback(loop_vars)
                self._cur_block.terminator = BrTerm(inc_bb.label)

            # Emit increment (may be empty: for(init; cond;))
            self._cur_block = inc_bb
            saved_pos = self._pos
            self._pos = inc_start
            if self._toks[inc_start].kind != TokKind.RPAREN:
                self._parse_assign_expr()
                while self._match(TokKind.COMMA):
                    self._parse_assign_expr()
            self._pos = saved_pos

            # Write back any variables modified by the increment expression
            # (or by direct-continue paths that bypassed the body writeback above).
            for var_name, init_val in loop_vars.items():
                cur_val = self._variables.get(var_name)
                if cur_val is not None and cur_val is not init_val and isinstance(cur_val, Value):
                    self._emit(BinInst(init_val, BinOp.ADD, cur_val, Const(init_val.ty, 0)))
                    self._variables[var_name] = init_val

            inc_bb.terminator = BrTerm(cond_bb.label)
            self._cur_block = exit_bb
            return

        # While loop
        if tok.kind == TokKind.KW_WHILE:
            self._advance()
            self._expect(TokKind.LPAREN)

            # Snapshot variables BEFORE parsing condition.
            # cond_bb instructions will reference these canonical Values.
            # At the end of each body iteration we write updated values back
            # to these canonical registers so cond_bb sees them on re-entry.
            while_entry_vars = dict(self._variables)

            cond_bb = self._new_block("while_cond")
            body_bb = self._new_block("while_body")
            exit_bb = self._new_block("while_exit")

            self._cur_block.terminator = BrTerm(cond_bb.label)
            self._cur_block = cond_bb
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(None)
            self._continue_targets.append(cond_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()

            # Write back any variables modified in the body to their canonical
            # cond-entry Values, then restore _variables to canonical state so
            # that code after the loop sees the correct (updated) registers.
            self._loop_writeback(while_entry_vars)

            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(cond_bb.label)

            self._cur_block = exit_bb
            return

        # do/while loop
        if tok.kind == TokKind.KW_DO:
            self._advance()

            # Snapshot before body so we know the canonical Values that the
            # condition block will reference after writeback.
            do_entry_vars = dict(self._variables)

            body_bb = self._new_block("do_body")
            cond_bb = self._new_block("do_cond")
            exit_bb = self._new_block("do_exit")

            self._cur_block.terminator = BrTerm(body_bb.label)
            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(None)
            self._continue_targets.append(cond_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()

            # Write back modified variables before parsing the condition so
            # that cond_bb's CmpInst references the canonical (writeback)
            # registers rather than pinning to the first-body Values.
            self._loop_writeback(do_entry_vars)

            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(cond_bb.label)

            self._cur_block = cond_bb
            self._expect(TokKind.KW_WHILE)
            self._expect(TokKind.LPAREN)
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)
            self._expect(TokKind.SEMI)
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = exit_bb
            return

        # break
        if tok.kind == TokKind.KW_BREAK:
            self._advance()
            self._expect(TokKind.SEMI)
            if not self._break_targets:
                raise ParseError("break outside of loop")
            # If the enclosing scope has a vars snapshot (switch or loop),
            # write back any modifications to canonical entry registers.
            if self._break_snapshots and self._break_snapshots[-1] is not None:
                self._loop_writeback(self._break_snapshots[-1])
            self._cur_block.terminator = BrTerm(self._break_targets[-1])
            # Dead code after break — create a new unreachable block
            self._cur_block = self._new_block("after_break")
            return

        # continue
        if tok.kind == TokKind.KW_CONTINUE:
            self._advance()
            self._expect(TokKind.SEMI)
            if not self._continue_targets:
                raise ParseError("continue outside of loop")
            self._cur_block.terminator = BrTerm(self._continue_targets[-1])
            self._cur_block = self._new_block("after_continue")
            return

        # switch/case
        if tok.kind == TokKind.KW_SWITCH:
            self._advance()
            self._expect(TokKind.LPAREN)
            switch_val = self._parse_expr()
            self._expect(TokKind.RPAREN)
            self._expect(TokKind.LBRACE)

            # Save current block so we can connect it to switch_dispatch later.
            pre_switch_bb = self._cur_block
            # Snapshot variables so each case starts from the same state and
            # modifications can be written back to canonical registers at break.
            vars_before_switch = dict(self._variables)

            exit_bb = self._new_block("switch_exit")
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(vars_before_switch)

            # Collect cases, resetting variable state for each new case/default
            # so each body starts from the same canonical environment.
            cases = []  # (value, case_bb)
            default_bb = None
            # When a case body has no break (fallthrough), remember the block so
            # we can connect it to the NEXT case block instead of exit_bb.
            _fallthrough_from = None
            while not self._match(TokKind.RBRACE):
                if self._peek().kind in (TokKind.KW_CASE, TokKind.KW_DEFAULT):
                    # If the block we're currently in has no terminator it is
                    # either a fallthrough case body or the pre_switch_bb.
                    # Guard: never close pre_switch_bb here — its terminator is
                    # set later to point at the switch_dispatch block.
                    if self._cur_block.terminator is None and self._cur_block is not pre_switch_bb:
                        # Fallthrough: writeback modified vars to canonical regs
                        # (so the next case sees the updated values via the
                        # same canonical entry Values), then remember this block
                        # so we can point it at the next case_bb below.
                        self._loop_writeback(vars_before_switch)
                        _fallthrough_from = self._cur_block
                    else:
                        _fallthrough_from = None
                    if self._peek().kind == TokKind.KW_CASE:
                        self._advance()
                        case_val = self._parse_expr()
                        self._expect(TokKind.COLON)
                        case_bb = self._new_block("case")
                        cases.append((case_val, case_bb))
                        if _fallthrough_from is not None:
                            _fallthrough_from.terminator = BrTerm(case_bb.label)
                            _fallthrough_from = None
                        self._variables = dict(vars_before_switch)
                        self._cur_block = case_bb
                    else:
                        self._advance()
                        self._expect(TokKind.COLON)
                        default_bb = self._new_block("default")
                        if _fallthrough_from is not None:
                            _fallthrough_from.terminator = BrTerm(default_bb.label)
                            _fallthrough_from = None
                        self._variables = dict(vars_before_switch)
                        self._cur_block = default_bb
                else:
                    self._parse_stmt()

            self._break_targets.pop()
            self._break_snapshots.pop()

            # Close the last active block (e.g. after_break stub left behind by
            # the final break statement, or a fall-through body ending at '}'.
            if self._cur_block.terminator is None:
                self._loop_writeback(vars_before_switch)
                self._cur_block.terminator = BrTerm(exit_bb.label)

            # Ensure all case/default blocks have a terminator.
            # If the body ended without a break (fall-through), write back and
            # fall to exit.
            for _, case_bb in cases:
                if case_bb.terminator is None:
                    self._cur_block = case_bb
                    self._loop_writeback(vars_before_switch)
                    case_bb.terminator = BrTerm(exit_bb.label)
            if default_bb and default_bb.terminator is None:
                self._cur_block = default_bb
                self._loop_writeback(vars_before_switch)
                default_bb.terminator = BrTerm(exit_bb.label)

            # Build comparison chain dispatch and connect the pre-switch block.
            entry_bb = self._new_block("switch_dispatch")
            if pre_switch_bb.terminator is None:
                pre_switch_bb.terminator = BrTerm(entry_bb.label)

            cur = entry_bb
            for case_val, case_bb in cases:
                cmp = self._new_val("cmp", ScalarTy(ScalarType.BOOL))
                self._cur_block = cur
                self._emit(CmpInst(cmp, CmpOp.EQ, switch_val, case_val))
                next_bb = self._new_block("switch_next")
                cur.terminator = CondBrTerm(cmp, case_bb.label, next_bb.label)
                cur = next_bb

            # Final dispatch: fall to default or exit
            self._cur_block = cur
            if default_bb:
                cur.terminator = BrTerm(default_bb.label)
            else:
                cur.terminator = BrTerm(exit_bb.label)

            # Restore canonical variable state at the exit block
            self._variables = dict(vars_before_switch)
            self._cur_block = exit_bb
            return

        # Return
        if tok.kind == TokKind.KW_RETURN:
            self._advance()
            if self._at(TokKind.SEMI):
                self._advance()
                ret_val = None
            else:
                ret_val = self._parse_expr()
                self._expect(TokKind.SEMI)
            if self._inline_return_target is not None:
                # Inside inlined device function — store return value and branch to merge
                return_dest, return_merge_label = self._inline_return_target
                if ret_val is not None:
                    zero = Const(return_dest.ty, 0.0 if (isinstance(return_dest.ty, ScalarTy) and return_dest.ty.is_float) else 0)
                    self._emit(BinInst(return_dest, BinOp.ADD, ret_val, zero))
                self._cur_block.terminator = BrTerm(return_merge_label)
                # Create dead block for any code after this return
                self._cur_block = self._new_block("after_inline_return")
            else:
                self._cur_block.terminator = RetTerm()
            return

        # Expression statement — check for array assignment: ptr[idx] = expr;
        # Capture original variable name before the LHS expression consumes the token
        # and may resolve it to an aliased Value with a different .name.
        _stmt_lhs_name = None
        if self._at(TokKind.IDENT):
            cand = self._peek().value
            next_pos = self._pos + 1
            if (next_pos < len(self._toks)
                    and self._toks[next_pos].kind in (
                        TokKind.ASSIGN, TokKind.PLUS_EQ, TokKind.MINUS_EQ,
                        TokKind.STAR_EQ)
                    and cand in self._variables):
                _stmt_lhs_name = cand
            elif (next_pos < len(self._toks)
                    and self._toks[next_pos].kind == TokKind.DOT
                    and cand in self._variables
                    and isinstance(self._variables[cand].ty, StructTy)):
                # struct.field = ... — detect the compound per-field variable name
                # so that the assignment handler updates the variable instead of
                # emitting a store-through-pointer (which is wrong for PtrTy fields).
                field_pos = next_pos + 1
                if field_pos < len(self._toks) and self._toks[field_pos].kind == TokKind.IDENT:
                    field_name = self._toks[field_pos].value
                    assign_pos = field_pos + 1
                    _assign_ops = (TokKind.ASSIGN, TokKind.PLUS_EQ, TokKind.MINUS_EQ,
                                   TokKind.STAR_EQ, TokKind.SLASH_EQ, TokKind.PERCENT_EQ,
                                   TokKind.AMP_EQ, TokKind.PIPE_EQ, TokKind.CARET_EQ,
                                   TokKind.LSHIFT_EQ, TokKind.RSHIFT_EQ)
                    if (assign_pos < len(self._toks)
                            and self._toks[assign_pos].kind in _assign_ops):
                        compound_name = f"{cand}_{field_name}"
                        if compound_name in self._variables:
                            _stmt_lhs_name = compound_name
        saved_pos = self._pos
        lhs = self._parse_lvalue_or_expr()
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_expr()
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                # Distinguish pointer variable reassignment (p = new_ptr) from
                # memory store through pointer (ptr[i] = val or *ptr = val).
                # _stmt_lhs_name is set iff the statement started with IDENT = ...
                # AND the IDENT is tracked in _variables.  In that case, treat as
                # variable update; otherwise emit a memory store.
                update_name = _stmt_lhs_name or lhs.name
                if _stmt_lhs_name and update_name in self._variables:
                    # Pointer variable reassignment: p = new_ptr_value
                    self._variables[update_name] = rhs
                else:
                    # Memory store through pointer: ptr[i] = val or *ptr = val
                    # Coerce rhs to the pointer's pointee type if there's a mismatch
                    if (isinstance(rhs, Value) and isinstance(rhs.ty, ScalarTy)
                            and isinstance(lhs.ty.pointee, ScalarTy)
                            and rhs.ty != lhs.ty.pointee):
                        coerced = self._new_val("coerce", lhs.ty.pointee)
                        self._emit(CvtInst(coerced, rhs))
                        rhs = coerced
                    self._emit(StoreInst(addr=lhs, value=rhs))
            elif isinstance(lhs, Value):
                update_name = _stmt_lhs_name or lhs.name
                if update_name in self._variables:
                    self._variables[update_name] = rhs
            # Comma-separated assignments: a = 0, b = 0; (for-loop init style)
            while self._match(TokKind.COMMA):
                lhs2_name = self._peek().value if self._at(TokKind.IDENT) else None
                lhs2 = self._parse_lvalue_or_expr()
                if self._match(TokKind.ASSIGN):
                    rhs2 = self._parse_assign_expr()
                    if isinstance(lhs2, Value) and not isinstance(lhs2.ty, PtrTy):
                        uname2 = lhs2_name or lhs2.name
                        if uname2 in self._variables:
                            self._variables[uname2] = rhs2
            self._expect(TokKind.SEMI)
            return
        # Compound assignment: +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL),
                             (TokKind.SLASH_EQ, BinOp.DIV),
                             (TokKind.PERCENT_EQ, BinOp.MOD),
                             (TokKind.AMP_EQ, BinOp.AND),
                             (TokKind.PIPE_EQ, BinOp.OR),
                             (TokKind.CARET_EQ, BinOp.XOR),
                             (TokKind.LSHIFT_EQ, BinOp.SHL),
                             (TokKind.RSHIFT_EQ, BinOp.SHR)]:
            if self._match(tok_kind):
                rhs = self._parse_expr()
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                    # Array compound: load current, compute, store back
                    cur = self._new_val("cur", lhs.ty.pointee)
                    self._emit(LoadInst(cur, lhs))
                    result = self._new_val("compound", cur.ty)
                    self._emit(BinInst(result, op, cur, rhs))
                    self._emit(StoreInst(addr=lhs, value=result))
                else:
                    result = self._new_val("compound", lhs.ty if isinstance(lhs, Value) else INT32)
                    self._emit(BinInst(result, op, lhs, rhs))
                    update_name = _stmt_lhs_name or (lhs.name if isinstance(lhs, Value) else None)
                    if update_name and update_name in self._variables:
                        self._variables[update_name] = result
                self._expect(TokKind.SEMI)
                return
        # Postfix ++/-- as statement: ptr++; ptr--; scalar++; scalar--;
        for tok_kind, op in [(TokKind.PLUSPLUS, BinOp.ADD),
                             (TokKind.MINUSMINUS, BinOp.SUB)]:
            if self._match(tok_kind):
                if isinstance(lhs, Value):
                    if isinstance(lhs.ty, PtrTy):
                        # Pointer advance: increment by element size
                        step = lhs.ty.pointee.size if isinstance(lhs.ty.pointee, ScalarTy) else 1
                        new_ptr = self._new_val(f"{lhs.name}_inc", lhs.ty)
                        self._emit(BinInst(new_ptr, op, lhs, Const(UINT64, step)))
                        update_name = _stmt_lhs_name or lhs.name
                        if update_name in self._variables:
                            self._variables[update_name] = new_ptr
                    else:
                        new_val = self._new_val(f"{lhs.name}_inc", lhs.ty)
                        self._emit(BinInst(new_val, op, lhs, Const(lhs.ty, 1)))
                        update_name = _stmt_lhs_name or lhs.name
                        if update_name in self._variables:
                            self._variables[update_name] = new_val
                self._expect(TokKind.SEMI)
                return
        # Comma-separated expression statements: a++, b += 2 or a = 0, b = 0
        while self._match(TokKind.COMMA):
            self._parse_assign_expr()
        self._expect(TokKind.SEMI)

    def _parse_lvalue_or_expr(self) -> Operand:
        """Parse an expression that might be an lvalue (address for assignment).

        For ptr[index] and ptr[index].field chains, returns the ADDRESS
        (PtrTy) without loading the scalar, so callers can emit StoreInst.
        For other expressions, falls through to _parse_expr (returns a value).
        """
        tok = self._peek()
        # *ptr lvalue: *name = value; — return pointer for StoreInst
        if tok.kind == TokKind.STAR:
            next_tok = self._toks[self._pos + 1] if self._pos + 1 < len(self._toks) else None
            if (next_tok and next_tok.kind == TokKind.IDENT
                    and next_tok.value in self._variables):
                var = self._variables[next_tok.value]
                if isinstance(var.ty, PtrTy):
                    self._advance()  # consume '*'
                    self._advance()  # consume ident
                    return var  # return pointer; StoreInst will be emitted by caller
        if tok.kind == TokKind.IDENT:
            name = tok.value
            if name in self._variables:
                var = self._variables[name]
                if isinstance(var.ty, PtrTy):
                    self._advance()
                    # ptr->field lvalue: compute address for StoreInst
                    if (self._at(TokKind.ARROW)
                            and isinstance(var.ty.pointee, StructTy)):
                        self._advance()  # consume '->'
                        member = self._expect(TokKind.IDENT).value
                        sty = var.ty.pointee
                        field_off = sty.field_offset(member)
                        field_ty = sty.field_type(member)
                        addr = self._new_val("faddr", PtrTy(field_ty, var.ty.addr_space))
                        self._emit(BinInst(addr, BinOp.ADD, var, Const(INT32, field_off)))
                        # Chain further ->field or .field accesses
                        while (self._at(TokKind.DOT) or self._at(TokKind.ARROW)):
                            self._advance()
                            sub_member = self._expect(TokKind.IDENT).value
                            if isinstance(addr.ty, PtrTy) and isinstance(addr.ty.pointee, StructTy):
                                ssty = addr.ty.pointee
                                s_off = ssty.field_offset(sub_member)
                                s_ty = ssty.field_type(sub_member)
                                new_addr = self._new_val("faddr", PtrTy(s_ty, addr.ty.addr_space))
                                self._emit(BinInst(new_addr, BinOp.ADD, addr, Const(INT32, s_off)))
                                addr = new_addr
                            else:
                                break
                        return addr  # Return ADDRESS for StoreInst
                    if self._match(TokKind.LBRACKET):
                        index = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        # Multi-dim array: if a second '[' follows and this var has
                        # a recorded row stride, use row stride for first index.
                        addr = None
                        row_stride = self._array_row_strides.get(name)
                        if row_stride is not None and self._at(TokKind.LBRACKET):
                            idx_ty = index.ty if isinstance(index, Value) else INT32
                            scaled = self._new_val("scale", idx_ty)
                            self._emit(BinInst(scaled, BinOp.MUL, index, Const(idx_ty, row_stride)))
                            addr = self._new_val("addr", var.ty)
                            self._emit(BinInst(addr, BinOp.ADD, var, scaled))
                            # Consume second [j] and compute final address
                            self._advance()  # consume '['
                            index2 = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            elem_size = var.ty.pointee.size
                            if elem_size != 1:
                                idx2_ty = index2.ty if isinstance(index2, Value) else INT32
                                scaled2 = self._new_val("scale", idx2_ty)
                                self._emit(BinInst(scaled2, BinOp.MUL, index2, Const(idx2_ty, elem_size)))
                                index2 = scaled2
                            final_addr = self._new_val("addr", var.ty)
                            self._emit(BinInst(final_addr, BinOp.ADD, addr, index2))
                            addr = final_addr
                        else:
                            elem_size = var.ty.pointee.size
                            if elem_size != 1:
                                idx_ty = index.ty if isinstance(index, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, index, Const(idx_ty, elem_size)))
                                index = scaled
                            addr = self._new_val("addr", var.ty)
                            self._emit(BinInst(addr, BinOp.ADD, var, index))
                            # Pointer-to-pointer: T**[i][j] — load row pointer then index
                            if (self._at(TokKind.LBRACKET)
                                    and isinstance(var.ty.pointee, PtrTy)):
                                row_ptr_ty = var.ty.pointee  # PtrTy(element_ty, ...)
                                row_ptr = self._new_val("rowptr", row_ptr_ty)
                                self._emit(LoadInst(row_ptr, addr))
                                self._advance()  # consume '['
                                index2 = self._parse_expr()
                                self._expect(TokKind.RBRACKET)
                                elem2_size = row_ptr_ty.pointee.size
                                if elem2_size != 1:
                                    idx2_ty = index2.ty if isinstance(index2, Value) else INT32
                                    scaled2 = self._new_val("scale", idx2_ty)
                                    self._emit(BinInst(scaled2, BinOp.MUL, index2,
                                                       Const(idx2_ty, elem2_size)))
                                    index2 = scaled2
                                final_addr = self._new_val("addr", row_ptr_ty)
                                self._emit(BinInst(final_addr, BinOp.ADD, row_ptr, index2))
                                addr = final_addr
                        # Follow chained .field access (e.g. p[tid].pos.x)
                        # keeping the result as a pointer (address) so that
                        # the caller can emit a StoreInst / compound read-modify-write.
                        while (self._at(TokKind.DOT)
                               and isinstance(addr.ty, PtrTy)
                               and isinstance(addr.ty.pointee, StructTy)):
                            self._advance()  # consume DOT
                            member = self._expect(TokKind.IDENT).value
                            sty = addr.ty.pointee
                            field_off = sty.field_offset(member)
                            field_ty = sty.field_type(member)
                            new_addr = self._new_val("faddr", PtrTy(field_ty, addr.ty.addr_space))
                            self._emit(BinInst(new_addr, BinOp.ADD, addr, Const(INT32, field_off)))
                            addr = new_addr
                        return addr  # Return ADDRESS, not loaded value
                    # Plain pointer variable: p = ... or p used as rvalue
                    return var
        # Fall back to normal expression parsing
        return self._parse_expr()

    def _parse_stmt_or_block(self):
        if self._match(TokKind.LBRACE):
            while not self._match(TokKind.RBRACE):
                self._parse_stmt()
        else:
            self._parse_stmt()

    # -- Top-level parsing ---------------------------------------------------

    def _parse_kernel(self):
        self._expect(TokKind.KW_GLOBAL)
        # Skip optional __launch_bounds__(maxThreads, minBlocks) — may appear
        # before or after the return type: both positions are valid CUDA.
        def _skip_launch_bounds(self):
            if self._at(TokKind.IDENT) and self._peek().value == '__launch_bounds__':
                self._advance()
                self._expect(TokKind.LPAREN)
                depth = 1
                while depth > 0:
                    if self._peek().kind == TokKind.LPAREN: depth += 1
                    if self._peek().kind == TokKind.RPAREN: depth -= 1
                    self._advance()
        _skip_launch_bounds(self)
        ret_ty = self._parse_type()  # should be void
        _skip_launch_bounds(self)    # also handle __global__ void __launch_bounds__(...)
        name = self._expect(TokKind.IDENT).value

        # Parameters
        self._expect(TokKind.LPAREN)
        params = []
        # (void) means no parameters
        if self._at(TokKind.KW_VOID) and self._toks[self._pos + 1].kind == TokKind.RPAREN:
            self._advance()  # consume 'void'
        if not self._at(TokKind.RPAREN):
            while True:
                pty = self._parse_type_with_ptr()
                pname = self._expect(TokKind.IDENT).value
                params.append(KernelParam(pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

        self._kernel = Kernel(name=name, params=params)
        self._variables = {}
        self._block_count = 0

        # Load kernel parameters into variables
        entry = self._new_block("entry")
        self._cur_block = entry
        self._lazy_params = {}
        for i, p in enumerate(params):
            val = self._new_val(p.name, p.ty)
            self._emit(ParamInst(val, i, p.name))
            self._variables[p.name] = val

        # Inject module-level extern __shared__ declarations into this kernel's scope.
        for shname, shty, shcount in self._module_shared_decls:
            smem_ty = PtrTy(shty, AddrSpace.SHARED)
            val = self._new_val(shname, smem_ty)
            self._variables[shname] = val
            if not hasattr(self._kernel, '_shared_decls'):
                self._kernel._shared_decls = []
            self._kernel._shared_decls.append((shname, shty, shcount))

        # Parse body
        self._expect(TokKind.LBRACE)
        while not self._match(TokKind.RBRACE):
            self._parse_stmt()

        # Ensure terminator
        if self._cur_block.terminator is None:
            self._cur_block.terminator = RetTerm()

        return self._kernel

    def _parse_struct_def(self):
        """Parse: struct/union [Name] { type field; ... } [;]
        Name is optional for anonymous structs used in typedefs.
        Union fields are treated as struct fields (all same offset model —
        type-punning won't work in PTX but code will compile).
        """
        # Accept both struct and union keywords
        if not self._match(TokKind.KW_STRUCT):
            self._expect(TokKind.KW_UNION)
        # Optional tag name — anonymous structs have none
        if self._at(TokKind.IDENT):
            name = self._advance().value
        else:
            name = f'__anon_{self._block_count}'
            self._block_count += 1
        if not self._at(TokKind.LBRACE):
            # Forward reference / usage without body: struct Name var;
            if name in self._struct_types:
                return self._struct_types[name]
            raise ParseError(f"Line {self._peek().line}: undefined struct '{name}'")
        self._expect(TokKind.LBRACE)
        fields = []
        field_arrays = {}  # base_name -> count for array members
        while not self._at(TokKind.RBRACE):
            # Skip C++ access specifiers: public:, private:, protected:
            if (self._at(TokKind.IDENT)
                    and self._peek().value in ('public', 'private', 'protected')
                    and self._pos + 1 < len(self._toks)
                    and self._toks[self._pos + 1].kind == TokKind.COLON):
                self._advance()  # consume 'public'/'private'/'protected'
                self._advance()  # consume ':'
                continue
            fty = self._parse_type_with_ptr()
            # Handle comma-separated field names: float x, y, z;
            while True:
                fname = self._expect(TokKind.IDENT).value
                # Inline array member: float data[N] or float m[R][C] — expand to
                # N (or R*C) scalar fields so struct layout accounts for full byte span.
                array_count = 1
                while self._match(TokKind.LBRACKET):
                    sz_op = self._parse_assign_expr()
                    dim = int(sz_op.value) if isinstance(sz_op, Const) else 1
                    array_count *= dim
                    self._expect(TokKind.RBRACKET)
                # Bitfield: field : width — consume and ignore width
                if self._match(TokKind.COLON):
                    self._parse_assign_expr()  # consume width expression
                if array_count > 1:
                    for k in range(array_count):
                        fields.append((f"{fname}_{k}", fty))
                    field_arrays[fname] = array_count
                else:
                    fields.append((fname, fty))
                if not self._match(TokKind.COMMA):
                    break
            self._expect(TokKind.SEMI)
        self._expect(TokKind.RBRACE)
        sty = StructTy(name, tuple(fields))
        self._struct_types[name] = sty
        if field_arrays:
            self._struct_field_arrays[name] = field_arrays
        # Only consume the trailing semicolon if not in a typedef context
        # (typedef will consume its own trailing ident + semi).
        # Check: if next token is SEMI or EOF, consume.
        if self._at(TokKind.SEMI):
            self._advance()
        return sty

    def _parse_enum_def(self):
        """Parse: enum [Name] { IDENT [= val] [, ...] } [;]
        Registers each enumerator as a module-level INT32 constant.
        No IR instructions are emitted — enum values are folded at parse time.
        """
        self._expect(TokKind.KW_ENUM)
        # Optional tag name — register as INT32 typedef so it can be used in
        # variable declarations and casts: Direction d; (Direction)(val)
        if self._at(TokKind.IDENT) and not self._at(TokKind.LBRACE):
            enum_tag = self._peek().value
            self._advance()
            self._typedefs[enum_tag] = INT32
        self._expect(TokKind.LBRACE)
        counter = 0
        while not self._at(TokKind.RBRACE):
            name = self._expect(TokKind.IDENT).value
            if self._match(TokKind.ASSIGN):
                val_const = self._parse_assign_expr()
                if isinstance(val_const, Const):
                    counter = int(val_const.value)
                else:
                    counter = 0  # non-const enum initializer: use 0 as fallback
            # Store as a compile-time constant (no IR emission)
            self._global_consts[name] = Const(INT32, counter)
            counter += 1
            if not self._match(TokKind.COMMA):
                break
        self._expect(TokKind.RBRACE)
        self._match(TokKind.SEMI)  # optional trailing semicolon

    def _parse_typedef(self):
        """Parse: typedef struct [Name] { ... } Alias;  or  typedef type name;"""
        self._expect(TokKind.KW_TYPEDEF)
        if self._at(TokKind.KW_ENUM):
            # typedef enum [Tag] { ... } Alias;
            self._parse_enum_def()     # registers enumerators as global consts
            # Alias is just another name for INT32
            if self._at(TokKind.IDENT):
                alias = self._advance().value
                self._typedefs[alias] = INT32
            self._match(TokKind.SEMI)
            return
        if self._at(TokKind.KW_STRUCT) or self._at(TokKind.KW_UNION):
            sty = self._parse_struct_def()
            # _parse_struct_def may or may not have consumed the ';'.
            # If there is an IDENT next, it is the typedef alias name.
            if self._at(TokKind.IDENT):
                alias = self._advance().value
                self._typedefs[alias] = sty
                self._match(TokKind.SEMI)
            else:
                # No alias — just register by struct name
                self._typedefs[sty.name] = sty
        else:
            ty = self._parse_type_with_ptr()
            # Function pointer typedef: typedef ret_ty (*alias)(params...);
            if self._at(TokKind.LPAREN):
                self._advance()          # consume '('
                self._match(TokKind.STAR)  # optional '*'
                alias = self._advance().value if self._at(TokKind.IDENT) else None
                self._match(TokKind.RPAREN)
                # Skip parameter list
                if self._at(TokKind.LPAREN):
                    depth = 0
                    while not self._at(TokKind.EOF):
                        if self._peek().kind == TokKind.LPAREN:
                            depth += 1; self._advance()
                        elif self._peek().kind == TokKind.RPAREN:
                            depth -= 1; self._advance()
                            if depth == 0:
                                break
                        else:
                            self._advance()
                self._match(TokKind.SEMI)
                if alias:
                    self._typedefs[alias] = ty  # map alias → return type (approximate)
                return
            alias = self._expect(TokKind.IDENT).value
            # Array typedef: typedef float arr_t[N]; — consume and ignore dimension
            if self._at(TokKind.LBRACKET):
                self._advance()
                if not self._at(TokKind.RBRACKET):
                    self._parse_assign_expr()
                self._expect(TokKind.RBRACKET)
            self._expect(TokKind.SEMI)
            self._typedefs[alias] = ty

    def _parse_device_func(self):
        """Parse __device__ function and store for inlining."""
        self._expect(TokKind.KW_DEVICE)
        # Consume any additional qualifiers: __device__ __forceinline__, __noinline__, etc.
        _FUNC_QUALIFIERS = {'__noinline__', '__forceinline__', '__inline__', '__attribute__',
                            '__cdecl__', '__stdcall__', '__fastcall__'}
        while (self._at(TokKind.KW_DEVICE) or self._at(TokKind.KW_STATIC)
               or (self._at(TokKind.IDENT) and self._peek().value in _FUNC_QUALIFIERS)):
            self._advance()
        ret_ty = self._parse_type_with_ptr()
        name = self._expect(TokKind.IDENT).value

        self._expect(TokKind.LPAREN)
        params = []
        # (void) means no parameters
        if self._at(TokKind.KW_VOID) and self._toks[self._pos + 1].kind == TokKind.RPAREN:
            self._advance()  # consume 'void'
        if not self._at(TokKind.RPAREN):
            while True:
                pty = self._parse_type_with_ptr()
                pname = self._expect(TokKind.IDENT).value
                params.append((pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

        # Forward declaration (prototype): __device__ float f(int x); — no body.
        if self._match(TokKind.SEMI):
            return  # nothing to register yet; definition will follow

        # Save token range for the body to replay during inlining
        body_start = self._pos
        # Skip the body
        self._expect(TokKind.LBRACE)
        depth = 1
        while depth > 0:
            if self._peek().kind == TokKind.LBRACE: depth += 1
            if self._peek().kind == TokKind.RBRACE: depth -= 1
            self._advance()
        body_end = self._pos

        self._device_funcs[name] = {
            'name': name,
            'ret_ty': ret_ty,
            'params': params,
            'body_start': body_start,
            'body_end': body_end,
        }

    def _parse_constant_decl(self, mod):
        """Parse: __constant__ type name[N]; or __constant__ type name;
        Registers the name as a module-level global constant pointer visible
        in all kernels. In PTX, __constant__ arrays are .const globals.
        """
        self._expect(TokKind.KW_CONSTANT)
        # Optional qualifiers (including __device__ in reversed qualifier order)
        while (self._at(TokKind.KW_CONST) or self._at(TokKind.KW_STATIC)
               or self._at(TokKind.KW_DEVICE)):
            self._advance()
        ty = self._parse_type_with_ptr()
        name = self._expect(TokKind.IDENT).value
        # Array declaration: __constant__ float arr[N];
        size = 1
        if self._match(TokKind.LBRACKET):
            sz_op = self._parse_assign_expr()
            size = int(sz_op.value) if isinstance(sz_op, Const) else 1
            self._expect(TokKind.RBRACKET)
        # Optional initializer: = { ... } or = expr — skip for module-level constants
        if self._match(TokKind.ASSIGN):
            depth = 0
            while not self._at(TokKind.EOF):
                if self._peek().kind == TokKind.LBRACE:
                    depth += 1; self._advance()
                elif self._peek().kind == TokKind.RBRACE:
                    depth -= 1; self._advance()
                    if depth == 0:
                        break
                elif depth == 0 and self._peek().kind == TokKind.SEMI:
                    break
                else:
                    self._advance()
        self._match(TokKind.SEMI)
        # Register as a module-level constant pointer (AddrSpace.CONST or GLOBAL)
        # Kernels that reference this name get a pointer Value they can index into.
        ptr_ty = PtrTy(ty, AddrSpace.CONST)
        self._global_consts[name] = SymbolRef(name, ptr_ty)
        mod.global_vars.append((name, ty, size, AddrSpace.CONST))

    def parse_module(self) -> Module:
        mod = Module()
        self._device_funcs = {}

        while not self._at(TokKind.EOF):
            # Skip leading storage class qualifiers (static, inline) before __global__/__device__
            while self._at(TokKind.KW_STATIC):
                self._advance()
            # Skip C++ namespace IDENT { ... } — treat contents as module-level
            if self._at(TokKind.IDENT) and self._peek().value == 'namespace':
                self._advance()  # consume 'namespace'
                if self._at(TokKind.IDENT):
                    self._advance()  # consume namespace name
                if self._at(TokKind.LBRACE):
                    self._advance()  # consume '{' — contents parsed normally; '}' consumed by fallback
                continue
            # Skip C++ template<...> prefix — we treat templates as plain functions
            if self._at(TokKind.IDENT) and self._peek().value == 'template':
                self._advance()  # consume 'template'
                if self._at(TokKind.LT):
                    self._advance()
                    depth = 1
                    while depth > 0 and not self._at(TokKind.EOF):
                        if self._peek().kind == TokKind.LT: depth += 1
                        elif self._peek().kind == TokKind.GT: depth -= 1
                        self._advance()
                    # Register all identifiers inside <...> as template-param typedefs → float
                    # (already consumed; just continue to parse the following function)
                continue
            if self._at(TokKind.KW_GLOBAL):
                mod.kernels.append(self._parse_kernel())
            elif self._at(TokKind.KW_DEVICE):
                # Peek ahead: if '(' appears before ';' it's a function; otherwise
                # it's a global device variable (__device__ int counter;).
                lookahead = self._pos + 1
                is_func = False
                while lookahead < len(self._toks):
                    k = self._toks[lookahead].kind
                    if k == TokKind.LPAREN:
                        is_func = True
                        break
                    if k in (TokKind.SEMI, TokKind.EOF):
                        break
                    lookahead += 1
                if is_func:
                    self._parse_device_func()
                else:
                    # Module-level __device__ variable: consume __device__ + qualifiers,
                    # parse the declaration, register as a global symbol ref.
                    is_const = False
                    while self._at(TokKind.KW_DEVICE) or self._at(TokKind.KW_STATIC):
                        self._advance()
                    if self._at(TokKind.KW_CONSTANT):
                        is_const = True
                        self._advance()
                    ty = self._parse_type_with_ptr()
                    name = self._expect(TokKind.IDENT).value
                    # Optional array size [N]
                    count = 1
                    if self._match(TokKind.LBRACKET):
                        sz_op = self._parse_assign_expr()
                        count = int(sz_op.value) if isinstance(sz_op, Const) else 1
                        self._expect(TokKind.RBRACKET)
                    # Optional initializer: = { ... } or = expr — skip for module-level
                    if self._match(TokKind.ASSIGN):
                        depth = 0
                        while not self._at(TokKind.EOF):
                            if self._peek().kind == TokKind.LBRACE:
                                depth += 1; self._advance()
                            elif self._peek().kind == TokKind.RBRACE:
                                depth -= 1; self._advance()
                                if depth == 0:
                                    break
                            elif depth == 0 and self._peek().kind == TokKind.SEMI:
                                break
                            else:
                                self._advance()
                    self._match(TokKind.SEMI)
                    addr = AddrSpace.CONST if is_const else AddrSpace.GLOBAL
                    ptr_ty = PtrTy(ty, addr)
                    self._global_consts[name] = SymbolRef(name, ptr_ty)
                    mod.global_vars.append((name, ty, count, addr))
            elif self._at(TokKind.KW_SHARED):
                # Module-level extern __shared__ type name[]; — dynamic shared memory.
                # Register as a pending declaration injected into every kernel that follows.
                self._advance()
                ty = self._parse_type()
                name = self._expect(TokKind.IDENT).value
                self._expect(TokKind.LBRACKET)
                count = 0  # 0 = dynamic shared (extern)
                if not self._at(TokKind.RBRACKET):
                    sz_op = self._parse_assign_expr()
                    count = int(sz_op.value) if isinstance(sz_op, Const) else 1
                self._expect(TokKind.RBRACKET)
                self._match(TokKind.SEMI)
                self._module_shared_decls.append((name, ty, count))
            elif self._at(TokKind.KW_STRUCT):
                self._parse_struct_def()
            elif self._at(TokKind.KW_UNION):
                self._parse_struct_def()
            elif self._at(TokKind.KW_TYPEDEF):
                self._parse_typedef()
            elif self._at(TokKind.KW_ENUM):
                self._parse_enum_def()
            elif self._at(TokKind.KW_CONSTANT):
                self._parse_constant_decl(mod)
            else:
                self._advance()
        return mod


def parse(source: str) -> Module:
    """Parse CUDA-subset C source into an IR Module."""
    tokens = lex(source)
    return Parser(tokens).parse_module()
