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

from .lexer import Token, TokKind, lex
from ..ir.types import (Type, ScalarTy, PtrTy, AddrSpace, ScalarType, StructTy,
                         INT32, UINT32, FLOAT, VOID, INT64, UINT64, DOUBLE, HALF)
from ..ir.nodes import (Module, Kernel, KernelParam, BasicBlock,
                         Value, Const, Operand,
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
        self._break_targets: list[str] = []          # stack of break target labels
        self._break_snapshots: list[Optional[dict]] = []  # vars snapshot at break scope entry
        self._continue_targets: list[str] = []       # stack of continue target labels
        self._inline_return_target = None  # (return_dest_val, return_merge_label) or None
        # Module-level compile-time constants (enum values, etc.)
        # These are visible in all kernels as Const operands without IR instructions.
        self._global_consts: dict[str, Const] = {}

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
            elif self._match(TokKind.KW_LONG):
                if self._match(TokKind.KW_LONG):
                    return UINT64
            return UINT32
        elif tok.kind == TokKind.KW_LONG:
            self._advance()
            if self._match(TokKind.KW_LONG):
                return INT64
            return INT32  # treat 'long' as int32 for simplicity

        # Struct type
        if tok.kind == TokKind.KW_STRUCT:
            self._advance()
            sname = self._expect(TokKind.IDENT).value
            if sname in self._struct_types:
                return self._struct_types[sname]
            raise ParseError(f"Line {tok.line}: undefined struct '{sname}'")

        # Typedef'd type
        if tok.kind == TokKind.IDENT and tok.value in self._typedefs:
            self._advance()
            return self._typedefs[tok.value]

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
            if (next_pos < len(self._toks)
                    and self._toks[next_pos].kind in (
                        TokKind.ASSIGN, TokKind.PLUS_EQ, TokKind.MINUS_EQ,
                        TokKind.STAR_EQ, TokKind.SLASH_EQ, TokKind.PERCENT_EQ,
                        TokKind.AMP_EQ, TokKind.PIPE_EQ, TokKind.CARET_EQ,
                        TokKind.LSHIFT_EQ, TokKind.RSHIFT_EQ)
                    and cand in self._variables):
                _lhs_orig_name = cand

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
            dest = self._new_val("bitor", self._result_type(lhs, rhs))
            self._emit(BinInst(dest, BinOp.OR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitxor_expr(self) -> Operand:
        lhs = self._parse_bitand_expr()
        while self._match(TokKind.CARET):
            rhs = self._parse_bitand_expr()
            dest = self._new_val("bitxor", self._result_type(lhs, rhs))
            self._emit(BinInst(dest, BinOp.XOR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitand_expr(self) -> Operand:
        lhs = self._parse_cmp_expr()
        while self._match(TokKind.AMP):
            rhs = self._parse_cmp_expr()
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
                # Shift result type is the left operand's type (C §6.5.7).
                # The right operand's type does NOT affect the result type.
                lhs_ty = lhs.ty if isinstance(lhs, (Value, Const)) else INT32
                dest = self._new_val("shl", lhs_ty)
                self._emit(BinInst(dest, BinOp.SHL, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.RSHIFT):
                rhs = self._parse_add_expr()
                # Shift result type is the left operand's type (C §6.5.7).
                # Critical: int x >> unsigned y must stay INT32 → shr.s32 (arithmetic).
                # Using _result_type would incorrectly return UINT32 (unsigned wins),
                # producing shr.b32 (logical) and giving wrong results for negative x.
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
                dest = self._new_val("add", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.ADD, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.MINUS):
                rhs = self._parse_mul_expr()
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
                dest = self._new_val("mul", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.MUL, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.SLASH):
                rhs = self._parse_unary_expr()
                dest = self._new_val("div", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.DIV, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.PERCENT):
                rhs = self._parse_unary_expr()
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
                if name in self._variables:
                    var = self._variables[name]
                    if isinstance(var.ty, PtrTy):
                        saved = self._pos
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
                        # Not array index, restore
                        self._pos = saved
            # Generic fallback
            operand = self._parse_unary_expr()
            if isinstance(operand, Value) and isinstance(operand.ty, PtrTy):
                return operand
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
                # Struct member access
                elif isinstance(lhs, Value) and isinstance(lhs.ty, StructTy):
                    sty = lhs.ty
                    field_off = sty.field_offset(member)
                    field_ty = sty.field_type(member)
                    # Compute address: &lhs + field_offset
                    # For now, emit as a load from a computed offset
                    # (this assumes lhs is a pointer to the struct)
                    dest = self._new_val(f"{lhs.name}_{member}", field_ty)
                    # TODO: proper struct field access via pointer arithmetic
                    lhs = dest
                elif isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy) and isinstance(lhs.ty.pointee, StructTy):
                    sty = lhs.ty.pointee
                    field_off = sty.field_offset(member)
                    field_ty = sty.field_type(member)
                    # ptr->field: compute address and load
                    offset_val = self._new_val("foff", INT32)
                    self._emit(BinInst(offset_val, BinOp.ADD, lhs, Const(INT32, field_off)))
                    addr = self._new_val("faddr", PtrTy(field_ty, lhs.ty.addr_space))
                    self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                    dest = self._new_val(f"{member}", field_ty)
                    self._emit(LoadInst(dest, addr))
                    lhs = dest
            else:
                break
        return lhs

    def _parse_primary_expr(self) -> Operand:
        tok = self._peek()

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
            val = int(raw.rstrip('uUlL'), 0)
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
                                     '__brev', '__brevll')
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
            if name in self._lazy_params and name not in self._variables:
                idx, p = self._lazy_params[name]
                val = self._new_val(p.name, p.ty)
                self._emit(ParamInst(val, idx, p.name))
                self._variables[p.name] = val

            # Variable reference
            if name in self._variables:
                return self._variables[name]

            # Module-level compile-time constants (enum values, etc.)
            if name in self._global_consts:
                return self._global_consts[name]

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

        # __shared__ declaration: __shared__ type name[size];
        if tok.kind == TokKind.KW_SHARED:
            self._advance()
            ty = self._parse_type()
            name = self._expect(TokKind.IDENT).value
            self._expect(TokKind.LBRACKET)
            size_tok = self._expect(TokKind.INT_LIT)
            size = int(size_tok.value)
            self._expect(TokKind.RBRACKET)
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
        if tok.kind in (TokKind.KW_INT, TokKind.KW_UNSIGNED, TokKind.KW_FLOAT,
                        TokKind.KW_DOUBLE, TokKind.KW_VOID, TokKind.KW_LONG,
                        TokKind.KW_HALF):
            ty = self._parse_type_with_ptr()
            while True:
                # Each declarator may have its own pointer stars: int *a, b, *c;
                decl_ty = ty
                while self._match(TokKind.STAR):
                    decl_ty = PtrTy(decl_ty, AddrSpace.GLOBAL)
                name = self._expect(TokKind.IDENT).value

                # Local array declaration: type name[N];
                # Allocate in .local memory, expose as a pointer.
                if self._at(TokKind.LBRACKET):
                    self._advance()
                    size_operand = self._parse_assign_expr()
                    self._expect(TokKind.RBRACKET)
                    size = int(size_operand.value) if isinstance(size_operand, Const) else 1
                    arr_ty = PtrTy(decl_ty, AddrSpace.LOCAL)
                    val = self._new_val(name, arr_ty)
                    self._variables[name] = val
                    # Store local array info for codegen
                    if not hasattr(self._kernel, '_local_decls'):
                        self._kernel._local_decls = []
                    self._kernel._local_decls.append((name, decl_ty, size, val))
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
                        # If rhs type matches declared type, use rhs directly (no copy).
                        # If there is a same-width signedness mismatch (e.g. int x = uint_expr),
                        # insert a CvtInst so that the variable carries the declared type.
                        # This ensures pointer arithmetic uses cvt.s64.s32 (sign-extending)
                        # rather than cvt.u64.u32 (zero-extending) for int-typed indices.
                        rhs_ty = rhs.ty
                        if (isinstance(decl_ty, ScalarTy) and isinstance(rhs_ty, ScalarTy)
                                and decl_ty != rhs_ty
                                and decl_ty.size == rhs_ty.size
                                and not decl_ty.is_float and not rhs_ty.is_float):
                            # Same-width integer type mismatch: coerce to declared type
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

            # Emit increment
            self._cur_block = inc_bb
            saved_pos = self._pos
            self._pos = inc_start
            self._parse_expr()
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
            while not self._match(TokKind.RBRACE):
                if self._peek().kind in (TokKind.KW_CASE, TokKind.KW_DEFAULT):
                    # If the block we're currently in has no terminator (e.g. an
                    # after_break stub or a fall-through case body), close it.
                    # Guard: never close pre_switch_bb here — its terminator is
                    # set later (line 1045) to point at the switch_dispatch block.
                    # Without this guard, the first `case` keyword would close
                    # pre_switch_bb with BrTerm(exit_bb), bypassing the dispatch
                    # entirely and producing a switch that always falls through to
                    # exit without testing any case.
                    if self._cur_block.terminator is None and self._cur_block is not pre_switch_bb:
                        self._loop_writeback(vars_before_switch)
                        self._cur_block.terminator = BrTerm(exit_bb.label)
                    if self._peek().kind == TokKind.KW_CASE:
                        self._advance()
                        case_val = self._parse_expr()
                        self._expect(TokKind.COLON)
                        case_bb = self._new_block("case")
                        cases.append((case_val, case_bb))
                        self._variables = dict(vars_before_switch)
                        self._cur_block = case_bb
                    else:
                        self._advance()
                        self._expect(TokKind.COLON)
                        default_bb = self._new_block("default")
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
        saved_pos = self._pos
        lhs = self._parse_lvalue_or_expr()
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_expr()
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                # Coerce rhs to the pointer's pointee type if there's a scalar mismatch
                # (e.g. half value stored to float* must widen via cvt.f32.f16).
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
            self._expect(TokKind.SEMI)
            return
        # Compound assignment: +=, -=, *=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL)]:
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
        self._expect(TokKind.SEMI)

    def _parse_lvalue_or_expr(self) -> Operand:
        """Parse an expression that might be an lvalue (address for assignment).

        For ptr[index], returns the ADDRESS (PtrTy) without loading.
        For other expressions, returns the value normally.
        """
        tok = self._peek()
        if tok.kind == TokKind.IDENT:
            name = tok.value
            if name in self._variables:
                var = self._variables[name]
                if isinstance(var.ty, PtrTy):
                    self._advance()
                    if self._match(TokKind.LBRACKET):
                        index = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        elem_size = var.ty.pointee.size
                        if elem_size != 1:
                            idx_ty = index.ty if isinstance(index, Value) else INT32
                            scaled = self._new_val("scale", idx_ty)
                            self._emit(BinInst(scaled, BinOp.MUL, index, Const(idx_ty, elem_size)))
                            index = scaled
                        addr = self._new_val("addr", var.ty)
                        self._emit(BinInst(addr, BinOp.ADD, var, index))
                        return addr  # Return ADDRESS, not loaded value
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
        # Skip optional __launch_bounds__(maxThreads, minBlocks)
        if self._at(TokKind.IDENT) and self._peek().value == '__launch_bounds__':
            self._advance()
            self._expect(TokKind.LPAREN)
            depth = 1
            while depth > 0:
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            # Consumed the closing )
        ret_ty = self._parse_type()  # should be void
        name = self._expect(TokKind.IDENT).value

        # Parameters
        self._expect(TokKind.LPAREN)
        params = []
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

        # Parse body
        self._expect(TokKind.LBRACE)
        while not self._match(TokKind.RBRACE):
            self._parse_stmt()

        # Ensure terminator
        if self._cur_block.terminator is None:
            self._cur_block.terminator = RetTerm()

        return self._kernel

    def _parse_struct_def(self):
        """Parse: struct Name { type field; ... };"""
        self._expect(TokKind.KW_STRUCT)
        name = self._expect(TokKind.IDENT).value
        self._expect(TokKind.LBRACE)
        fields = []
        while not self._at(TokKind.RBRACE):
            fty = self._parse_type_with_ptr()
            fname = self._expect(TokKind.IDENT).value
            self._expect(TokKind.SEMI)
            fields.append((fname, fty))
        self._expect(TokKind.RBRACE)
        self._expect(TokKind.SEMI)
        sty = StructTy(name, tuple(fields))
        self._struct_types[name] = sty
        return sty

    def _parse_enum_def(self):
        """Parse: enum [Name] { IDENT [= val] [, ...] } [;]
        Registers each enumerator as a module-level INT32 constant.
        No IR instructions are emitted — enum values are folded at parse time.
        """
        self._expect(TokKind.KW_ENUM)
        # Optional tag name
        if self._at(TokKind.IDENT) and not self._at(TokKind.LBRACE):
            self._advance()  # consume tag, ignored
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
        """Parse: typedef struct Name Name;  or  typedef type name;"""
        self._expect(TokKind.KW_TYPEDEF)
        if self._at(TokKind.KW_STRUCT):
            sty = self._parse_struct_def()
            # typedef struct Foo Foo; — the name after } is the typedef alias
            # But we already consumed ;. Check if there's another ident.
            self._typedefs[sty.name] = sty
        else:
            ty = self._parse_type_with_ptr()
            alias = self._expect(TokKind.IDENT).value
            self._expect(TokKind.SEMI)
            self._typedefs[alias] = ty

    def _parse_device_func(self):
        """Parse __device__ function and store for inlining."""
        self._expect(TokKind.KW_DEVICE)
        # Consume any additional qualifiers: __device__ __forceinline__, inline __device__, etc.
        while self._at(TokKind.KW_DEVICE) or self._at(TokKind.KW_STATIC):
            self._advance()
        ret_ty = self._parse_type_with_ptr()
        name = self._expect(TokKind.IDENT).value

        self._expect(TokKind.LPAREN)
        params = []
        if not self._at(TokKind.RPAREN):
            while True:
                pty = self._parse_type_with_ptr()
                pname = self._expect(TokKind.IDENT).value
                params.append((pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

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

    def parse_module(self) -> Module:
        mod = Module()
        self._device_funcs = {}

        while not self._at(TokKind.EOF):
            # Skip leading storage class qualifiers (static, inline) before __global__/__device__
            while self._at(TokKind.KW_STATIC):
                self._advance()
            if self._at(TokKind.KW_GLOBAL):
                mod.kernels.append(self._parse_kernel())
            elif self._at(TokKind.KW_DEVICE):
                self._parse_device_func()
            elif self._at(TokKind.KW_STRUCT):
                self._parse_struct_def()
            elif self._at(TokKind.KW_TYPEDEF):
                self._parse_typedef()
            elif self._at(TokKind.KW_ENUM):
                self._parse_enum_def()
            else:
                self._advance()
        return mod


def parse(source: str) -> Module:
    """Parse CUDA-subset C source into an IR Module."""
    tokens = lex(source)
    return Parser(tokens).parse_module()
