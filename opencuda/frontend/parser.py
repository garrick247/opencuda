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
                         INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT, VOID,
                         INT64, UINT64, DOUBLE, HALF)
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
        self._continue_snapshots: list[Optional[dict]] = []  # vars snapshot at continue scope entry
        self._inline_return_target = None  # (return_dest_val, return_merge_label) or None
        # Maps return_dest.id → {field_name: Value} for struct-returning inlined functions.
        # Populated by the return handler; read by the struct declaration/assignment handler.
        self._inline_struct_return_fields: dict[int, dict] = {}
        # Multi-dimensional array row strides: maps var_name → row_stride_bytes.
        # For float tile[16][16], row_stride = 16*sizeof(float) = 64.
        # Used to compute tile[i][j] as *(tile + i*64 + j*4) rather than *(tile + i*4 + j*4).
        self._array_row_strides: dict[str, int] = {}
        # Module-level compile-time constants (enum values, etc.)
        # These are visible in all kernels as Const operands without IR instructions.
        self._global_consts: dict[str, Const] = {}
        # Kernel-scoped identity-copy chain: maps dest Value id → source Value for
        # every BinInst(dest, ADD, src, Const(0)) emitted in the current kernel.
        # Used by _loop_writeback to trace through materialization intermediates
        # and correctly sequence parallel copies even when the materialization
        # instructions live in a different block than the writeback block.
        # Reset at the start of each kernel.
        self._copy_chain_global: dict[int, "Value"] = {}
        # Module-level extern __shared__ declarations: list of (name, scalar_ty).
        # Injected into each kernel's scope at kernel parse time.
        self._module_shared_decls: list = []
        # Names declared as __shared__ scalar (no brackets): __shared__ float total;
        # These are PtrTy(SHARED) variables that must be auto-dereferenced on rvalue use
        # and must use StoreInst (not variable reassignment) on plain assignment.
        self._shared_scalars: set = set()
        # Block-scope tracking for variable shadowing: each entry is the set of
        # variable names declared in that block.  Push on '{', pop on '}'.
        # Used to restore outer bindings when an inner block re-declares a name.
        self._scope_locals_stack: list[set] = []
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

    # Keywords that are not C reserved words but are used as type names in CUDA;
    # they may legally appear as user-defined identifiers (variable/param names).
    _SOFT_KW_AS_IDENT: set = frozenset({TokKind.KW_HALF})

    def _expect_ident(self) -> Token:
        """Like _expect(IDENT) but also accepts soft keywords used as identifiers."""
        tok = self._peek()
        if tok.kind == TokKind.IDENT or tok.kind in self._SOFT_KW_AS_IDENT:
            return self._advance()
        raise ParseError(f"Line {tok.line}: expected IDENT, got {tok.kind.name} '{tok.value}'")

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
        # Track identity copies (dest := src + 0) for parallel-copy sequencing.
        # This allows _loop_writeback to trace through materialization intermediates
        # emitted in earlier blocks when ordering writeback copies.
        if (isinstance(inst, BinInst)
                and inst.op == BinOp.ADD
                and isinstance(inst.rhs, Const)
                and inst.rhs.value == 0
                and isinstance(inst.lhs, Value)
                and isinstance(inst.dest, Value)):
            self._copy_chain_global[inst.dest.id] = inst.lhs

    def _new_val(self, name: str, ty: Type) -> Value:
        return self._kernel.new_value(name, ty)

    @staticmethod
    def _const_fold(op: 'BinOp', lhs: Operand, rhs: Operand) -> 'Const | None':
        """Fold two compile-time constants into a single Const. Returns None if
        either operand is not Const or the operation is not constant-foldable."""
        if not isinstance(lhs, Const) or not isinstance(rhs, Const):
            return None
        a_is_float = isinstance(lhs.ty, ScalarTy) and lhs.ty.is_float
        b_is_float = isinstance(rhs.ty, ScalarTy) and rhs.ty.is_float
        is_float = a_is_float or b_is_float
        # Result type: prefer float over int; use lhs type as primary
        ty = lhs.ty if isinstance(lhs.ty, ScalarTy) else INT32
        if b_is_float and not a_is_float:
            ty = rhs.ty
        try:
            if is_float:
                a, b = float(lhs.value), float(rhs.value)
                result = {
                    BinOp.ADD: a + b,
                    BinOp.SUB: a - b,
                    BinOp.MUL: a * b,
                    BinOp.DIV: a / b if b != 0.0 else 0.0,
                    BinOp.MOD: a % b if b != 0.0 else 0.0,
                }.get(op)
            else:
                a, b = int(lhs.value), int(rhs.value)
                # C integer division truncates toward zero (not Python floor).
                # -7 // 2 = -4 in Python, but -7 / 2 = -3 in C.
                _cdiv = (lambda a, b: (abs(a) // abs(b)) * (1 if (a >= 0) == (b >= 0) else -1)) if b != 0 else (lambda a, b: 0)
                _cmod = (lambda a, b: (abs(a) - (abs(a) // abs(b)) * abs(b)) * (1 if a >= 0 else -1)) if b != 0 else (lambda a, b: 0)
                result = {
                    BinOp.ADD: a + b,
                    BinOp.SUB: a - b,
                    BinOp.MUL: a * b,
                    BinOp.DIV: _cdiv(a, b),
                    BinOp.MOD: _cmod(a, b),
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

    def _scale_ptr_arith_offset(self, ptr_ty: PtrTy, int_op: Operand) -> Operand:
        """Scale an integer operand by the pointee size for C pointer arithmetic.

        In C, ptr + n advances n * sizeof(*ptr) bytes.  The emitter emits
        raw byte-offset additions (add.u64), so the parser must multiply the
        integer operand by elem_size before building the BinInst.

        Returns the original operand unchanged when elem_size == 1 (char*).
        """
        elem_size = getattr(ptr_ty.pointee, 'size', 4)
        if elem_size == 1:
            return int_op
        int_ty = int_op.ty if isinstance(int_op, (Value, Const)) else INT32
        if isinstance(int_op, Const):
            return Const(int_ty, int_op.value * elem_size)
        scaled = self._new_val("ptr_off", int_ty)
        self._emit(BinInst(scaled, BinOp.MUL, int_op, Const(int_ty, elem_size)))
        return scaled

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

        Copy sequencing: when multiple variables are written back, a copy
        (entry_val_A := cur_val_A) must not be emitted before any other pending
        copy whose cur_val is entry_val_A — otherwise the first copy overwrites
        a source register that a later copy still needs to read.  Copies are
        reordered here to avoid that clobber (parallel-copy sequencing).
        """
        # --- Pass 1: struct sentinels — restore binding, no emit needed ---
        for var_name, entry_val in entry_vars.items():
            if not isinstance(entry_val, Value):
                continue
            if isinstance(entry_val.ty, StructTy):
                if self._variables.get(var_name) is not entry_val:
                    self._variables[var_name] = entry_val

        # --- Pass 2: collect pending scalar copies ---
        pending: list[tuple[str, "Value", object]] = []
        for var_name, entry_val in entry_vars.items():
            if not isinstance(entry_val, Value):
                continue
            if isinstance(entry_val.ty, StructTy):
                continue
            cur_val = self._variables.get(var_name)
            if cur_val is None or cur_val is entry_val:
                continue
            if isinstance(cur_val, Value):
                if isinstance(cur_val.ty, StructTy):
                    continue
                pending.append((var_name, entry_val, cur_val))
            elif isinstance(cur_val, Const):
                pending.append((var_name, entry_val, cur_val))

        # --- Pass 3: sequence copies to avoid clobbering ---
        # After optimization, identity copies (mat = V + 0) are folded so that
        # cur_val in the writeback refers to a materialized intermediate, not
        # directly to entry_val of another copy.  Trace through identity-copy
        # chains in the current block to find the true source of each cur_val;
        # use those root sources for conflict detection.
        #
        # Example: s.second = s.best; s.best = v; generates:
        #   (a) mat_second = s_best + 0   [materialization in if-body]
        #   (b) mat_best   = v + 0         [materialization in if-body]
        #   (c) writeback: s_best   := mat_best   [pending copy for s_best]
        #   (d) writeback: s_second := mat_second [pending copy for s_second]
        # After fold: mat_second → s_best, mat_best → v, leaving:
        #   (c') s_best   := v
        #   (d') s_second := s_best   ← reads s_best AFTER (c') clobbers it
        # Root-tracing detects that root(mat_second) is s_best = entry_val(s_best),
        # so (d) must come before (c).

        # Use the kernel-scoped identity-copy chain (populated in _emit across all
        # blocks) to trace through materialization intermediates.  This handles
        # the case where the materialization copy (mat = entry_val + 0) was emitted
        # in a prior block (e.g. for_body_6), not in the current writeback block.
        _chain = self._copy_chain_global

        def _root(v: object) -> object:
            """Follow identity-copy chain to the ultimate source."""
            seen: set[int] = set()
            while isinstance(v, Value) and v.id not in seen:
                seen.add(v.id)
                src = _chain.get(v.id)
                if src is None:
                    break
                v = src
            return v

        ordered: list[tuple[str, "Value", object]] = []
        remaining = list(pending)
        guard = len(remaining) * len(remaining) + len(remaining) + 1
        iters = 0
        while remaining:
            iters += 1
            if iters > guard:
                break
            safe_idx = None
            for i, (_, ev_i, _) in enumerate(remaining):
                # Safe to emit if no other remaining copy's rooted cur_val traces
                # to ev_i (which would be clobbered by emitting this copy first)
                if not any(
                    _root(cv_j) is ev_i
                    for j, (_, _, cv_j) in enumerate(remaining) if j != i
                ):
                    safe_idx = i
                    break
            if safe_idx is None:
                # Cycle: emit first (cycles in struct-field writebacks are
                # vanishingly rare and would require a temp to fix properly)
                safe_idx = 0
            ordered.append(remaining.pop(safe_idx))

        # --- Pass 4: emit in safe order ---
        for var_name, entry_val, cur_val in ordered:
            self._emit(BinInst(entry_val, BinOp.ADD, cur_val, Const(entry_val.ty, 0)))
            self._variables[var_name] = entry_val

    # -- Type parsing --------------------------------------------------------

    def _parse_type(self) -> Type:
        """Parse a C type specifier."""
        # Skip leading qualifiers that don't affect the IR type.
        while self._peek().kind in (TokKind.KW_VOLATILE, TokKind.KW_CONST):
            self._advance()
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
                return UINT16
            elif self._match(TokKind.KW_CHAR):
                return UINT8
            elif self._match(TokKind.KW_LONG):
                if self._match(TokKind.KW_LONG):
                    self._match(TokKind.KW_INT)  # optional "unsigned long long int"
                    return UINT64
                self._match(TokKind.KW_INT)  # optional "unsigned long int"
            return UINT32
        elif tok.kind == TokKind.KW_LONG:
            self._advance()
            if self._match(TokKind.KW_LONG):
                self._match(TokKind.KW_INT)  # optional "long long int"
                return INT64
            self._match(TokKind.KW_INT)  # optional "long int"
            return INT32  # treat 'long' as int32 for simplicity (CUDA device code)
        elif tok.kind == TokKind.KW_SHORT:
            self._advance()
            self._match(TokKind.KW_INT)  # optional trailing 'int'
            return INT16
        elif tok.kind == TokKind.KW_SIGNED:
            self._advance()
            # signed [int | char | long | short] — preserve sub-word sizes
            if self._match(TokKind.KW_CHAR):
                return INT8
            elif self._match(TokKind.KW_SHORT):
                self._match(TokKind.KW_INT)
                return INT16
            elif self._match(TokKind.KW_LONG):
                if self._match(TokKind.KW_LONG):
                    self._match(TokKind.KW_INT)
                    return INT64
                self._match(TokKind.KW_INT)  # optional "signed long int"
                return INT32
            else:
                self._match(TokKind.KW_INT)
                return INT32
        elif tok.kind == TokKind.KW_CHAR:
            self._advance()
            return INT8
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

        # Enum type used as variable/parameter type: treat as INT32
        if tok.kind == TokKind.KW_ENUM:
            self._advance()  # consume 'enum'
            if self._at(TokKind.IDENT):
                self._advance()  # consume enum tag name (optional)
            return INT32

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
        # Track const/volatile on the pointee (before or immediately after the base type).
        self._match(TokKind.KW_STATIC)    # static/inline/register before type
        pointee_volatile = bool(self._match(TokKind.KW_VOLATILE))  # volatile before type
        pointee_const = self._match(TokKind.KW_CONST)
        if self._match(TokKind.KW_VOLATILE):  # volatile after const (e.g. const volatile T *)
            pointee_volatile = True
        self._match(TokKind.KW_STATIC)
        base = self._parse_type()
        # "float const *" or "float volatile *" — qualifier after base type
        if self._match(TokKind.KW_CONST):
            pointee_const = True
        if self._match(TokKind.KW_VOLATILE):
            pointee_volatile = True
        while self._match(TokKind.STAR):
            # "float * const" — const AFTER star means pointer-const, not pointee-const
            ptr_const = self._match(TokKind.KW_CONST)  # noqa: F841 (consumed, not used)
            self._match(TokKind.KW_VOLATILE)            # "float * volatile" — skip
            base = PtrTy(base, AddrSpace.GLOBAL, volatile=pointee_volatile)
            pointee_volatile = False  # volatile only applies to the innermost pointee
        # __restrict__ qualifier
        has_restrict = (self._at(TokKind.IDENT) and self._peek().value == '__restrict__')
        if has_restrict:
            self._advance()
        # Upgrade to CONST addr space when programmer guarantees read-only + no aliasing.
        # This allows codegen to emit ld.global.nc (non-caching / read-only cache).
        if pointee_const and has_restrict and isinstance(base, PtrTy):
            base = PtrTy(base.pointee, AddrSpace.CONST, volatile=base.volatile)
        return base

    # -- Expression parsing (precedence climbing) ----------------------------

    def _parse_expr(self) -> Operand:
        result = self._parse_assign_expr()
        while self._match(TokKind.COMMA):
            # Comma operator: evaluate left for side effects, discard its
            # result, evaluate right and return it.  Used in patterns like
            # while (i++, i < n) and general comma expressions.
            result = self._parse_assign_expr()
        return result

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
            result_ty = self._result_type(true_val, false_val)
            dest = self._new_val("ternary", result_ty)
            # General case: lower to branches
            true_bb = self._new_block("tern_true")
            false_bb = self._new_block("tern_false")
            merge_bb = self._new_block("tern_merge")
            self._cur_block.terminator = CondBrTerm(lhs, true_bb.label, false_bb.label)
            self._cur_block = true_bb
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
                # Exception: __shared__ scalar vars must always use StoreInst,
                # not variable rebinding, because the memory persists across threads.
                if (_lhs_orig_name
                        and _lhs_orig_name in self._variables
                        and _lhs_orig_name not in self._shared_scalars):
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
                # Materialize constants: if rhs is a Const and the previous binding
                # was a Value, emit a BinInst copy so the variable keeps a Value
                # binding.  This is critical for loop-carry: if struct fields are
                # assigned literal constants (acc.x = 0.0f), they must remain Values
                # in _variables so _loop_writeback can track and carry them across
                # iterations.  Mirrors the same materialization done for scalar
                # declarations with const initializers (int i = 0 → BinInst+Value).
                prev = self._variables[update_name]
                if (isinstance(prev, Value)
                        and isinstance(prev.ty, ScalarTy)
                        and isinstance(rhs.ty, ScalarTy)):
                    if (isinstance(rhs, Const)
                            or (isinstance(rhs, Value) and rhs.name != update_name)):
                        # Materialize: emit an identity copy so the target has its
                        # own dedicated register.  This is critical for loop-carry:
                        # if rhs is a different-named Value (e.g. s.count = n assigns
                        # the n param to s_count), the loop writeback would otherwise
                        # use the rhs register as the canonical entry, modifying the
                        # wrong register on writeback (e.g. decrementing n itself).
                        mat = self._new_val(update_name, prev.ty)
                        _zero = Const(prev.ty, 0.0 if prev.ty.is_float else 0)
                        self._emit(BinInst(mat, BinOp.ADD, rhs, _zero))
                        rhs = mat
                self._variables[update_name] = rhs
                # Struct-to-struct assignment: propagate per-field values so that
                # downstream field accesses (acc.tx, acc.ty, ...) see the new values.
                # This handles the case where _parse_lvalue_or_expr falls through to
                # _parse_expr for StructTy variables, meaning _parse_stmt's assignment
                # handler never runs.
                if isinstance(rhs, Value) and isinstance(rhs.ty, StructTy):
                    field_map = self._inline_struct_return_fields.get(rhs.id, {})
                    for _fname, _fty in rhs.ty.fields:
                        if not isinstance(_fty, ScalarTy):
                            continue
                        _src_f = field_map.get(_fname)
                        if _src_f is None:
                            _src_f = self._variables.get(f"{rhs.name}_{_fname}")
                        _dest_fkey = f"{update_name}_{_fname}"
                        if _src_f is not None and _dest_fkey in self._variables:
                            _new_fv = self._new_val(_dest_fkey, _fty)
                            self._emit(BinInst(
                                _new_fv, BinOp.ADD, _src_f,
                                Const(_fty, 0.0 if _fty.is_float else 0)))
                            self._variables[_dest_fkey] = _new_fv
                    # Restore the struct sentinel with the LHS variable name so
                    # subsequent "s.field" lookups build field_key = "s_field"
                    # (not "rhs_name_field" when rhs is an inline return sentinel).
                    cur_sent = self._variables.get(update_name)
                    if (isinstance(cur_sent, Value)
                            and isinstance(cur_sent.ty, StructTy)
                            and cur_sent.name != update_name):
                        fresh_sent = self._new_val(update_name, cur_sent.ty)
                        self._variables[update_name] = fresh_sent
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
            # Short-circuit: if LHS is true, skip RHS evaluation.
            # Generate control flow: if lhs { dest = 1 } else { dest = rhs }
            dest = self._new_val("lor", INT32)
            rhs_bb   = self._new_block("lor_rhs")
            skip_bb  = self._new_block("lor_skip")
            merge_bb = self._new_block("lor_merge")
            # LHS true → skip to lor_skip (result = 1), else → lor_rhs
            self._cur_block.terminator = CondBrTerm(lhs, skip_bb.label, rhs_bb.label)
            # RHS block: evaluate RHS, store in dest.
            # IMPORTANT: parsing RHS may create new blocks and change _cur_block.
            # Set the terminator on self._cur_block (final RHS block), not rhs_bb.
            # Snapshot variables so we can detect rebindings in the RHS and make
            # them multi-def by emitting default copies in lor_skip.
            vars_before_rhs = dict(self._variables)
            self._cur_block = rhs_bb
            rhs = self._parse_and_expr()
            self._emit(BinInst(dest, BinOp.ADD, rhs, Const(INT32, 0)))
            self._cur_block.terminator = BrTerm(merge_bb.label)
            # Collect variables rebound inside lor_rhs (assignment expressions, etc.)
            _rebound = [
                (new_v, vars_before_rhs[vname])
                for vname, new_v in self._variables.items()
                if (isinstance(new_v, Value)
                    and vname in vars_before_rhs
                    and vars_before_rhs[vname] is not new_v
                    and isinstance(vars_before_rhs[vname], Value))
            ]
            # Skip block: LHS was true, result is 1.
            # Also emit default copies for rebound vars to make them multi-def,
            # exempting them from the verifier's single-def dominance check.
            self._cur_block = skip_bb
            self._emit(BinInst(dest, BinOp.ADD, Const(INT32, 1), Const(INT32, 0)))
            for new_v, old_v in _rebound:
                _zero = Const(new_v.ty, 0.0 if isinstance(new_v.ty, ScalarTy) and new_v.ty.is_float else 0)
                self._emit(BinInst(new_v, BinOp.ADD, old_v, _zero))
            self._cur_block.terminator = BrTerm(merge_bb.label)
            self._cur_block = merge_bb
            lhs = dest
        return lhs

    def _parse_and_expr(self) -> Operand:
        lhs = self._parse_bitor_expr()
        while self._match(TokKind.AND):
            # Short-circuit: if LHS is false, skip RHS evaluation.
            # Generate control flow: if lhs { dest = rhs } else { dest = 0 }
            # This prevents OOB memory loads when the LHS guards an array access.
            dest = self._new_val("land", INT32)
            rhs_bb   = self._new_block("land_rhs")
            skip_bb  = self._new_block("land_skip")
            merge_bb = self._new_block("land_merge")
            # LHS true → land_rhs (eval RHS), LHS false → land_skip (result = 0)
            self._cur_block.terminator = CondBrTerm(lhs, rhs_bb.label, skip_bb.label)
            # RHS block: evaluate RHS, store in dest.
            # IMPORTANT: parsing RHS may create new blocks and change _cur_block
            # (e.g. if RHS contains || or another &&).  Set the terminator on
            # self._cur_block (the final block after RHS evaluation), not rhs_bb.
            # Snapshot variables so we can detect rebindings and make them multi-def.
            vars_before_rhs = dict(self._variables)
            self._cur_block = rhs_bb
            rhs = self._parse_bitor_expr()
            self._emit(BinInst(dest, BinOp.ADD, rhs, Const(INT32, 0)))
            self._cur_block.terminator = BrTerm(merge_bb.label)
            # Collect variables rebound inside land_rhs (assignment expressions, etc.)
            # E.g. `while (i < n && (v = arr[i]) != 0)` rebinds `v` in land_rhs.
            # Emit a copy of each rebound Value in land_skip so it is multi-defined.
            # Multi-def Values are exempt from the verifier's dominance check, and
            # the land_skip path never reaches the loop body so the default value
            # (old binding) is never observed.
            _rebound = [
                (new_v, vars_before_rhs[vname])
                for vname, new_v in self._variables.items()
                if (isinstance(new_v, Value)
                    and vname in vars_before_rhs
                    and vars_before_rhs[vname] is not new_v
                    and isinstance(vars_before_rhs[vname], Value))
            ]
            # Skip block: LHS was false, result is 0.
            self._cur_block = skip_bb
            self._emit(BinInst(dest, BinOp.ADD, Const(INT32, 0), Const(INT32, 0)))
            for new_v, old_v in _rebound:
                _zero = Const(new_v.ty, 0.0 if isinstance(new_v.ty, ScalarTy) and new_v.ty.is_float else 0)
                self._emit(BinInst(new_v, BinOp.ADD, old_v, _zero))
            self._cur_block.terminator = BrTerm(merge_bb.label)
            self._cur_block = merge_bb
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
                # Return BOOL type so 'int x = (a > b)' triggers CvtInst (selp.s32),
                # not a BinInst copy that identity_fold would propagate back to the
                # raw predicate register.
                dest = self._new_val("cmp", ScalarTy(ScalarType.BOOL))
                self._emit(CmpInst(dest, cmp_op, lhs, rhs))
                return dest
        return lhs

    def _result_type(self, a: Operand, b: Operand) -> Type:
        """Determine result type with promotion (wider float wins, float > int)."""
        a_ty = a.ty if isinstance(a, (Value, Const)) else FLOAT
        b_ty = b.ty if isinstance(b, (Value, Const)) else FLOAT
        # BOOL undergoes integer promotion to INT32 in arithmetic (C §6.3.1.1).
        _bool_ty = ScalarTy(ScalarType.BOOL)
        if a_ty == _bool_ty:
            a_ty = INT32
        if b_ty == _bool_ty:
            b_ty = INT32
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
                # C pointer arithmetic: ptr + n means ptr + n*sizeof(*ptr) bytes.
                lhs_ty = lhs.ty if isinstance(lhs, (Value, Const)) else None
                rhs_ty = rhs.ty if isinstance(rhs, (Value, Const)) else None
                if isinstance(lhs_ty, PtrTy) and not isinstance(rhs_ty, PtrTy):
                    rhs = self._scale_ptr_arith_offset(lhs_ty, rhs)
                elif isinstance(rhs_ty, PtrTy) and not isinstance(lhs_ty, PtrTy):
                    lhs = self._scale_ptr_arith_offset(rhs_ty, lhs)
                folded = self._const_fold(BinOp.ADD, lhs, rhs)
                if folded is not None:
                    lhs = folded
                else:
                    dest = self._new_val("add", self._result_type(lhs, rhs))
                    self._emit(BinInst(dest, BinOp.ADD, lhs, rhs))
                    lhs = dest
            elif self._match(TokKind.MINUS):
                rhs = self._parse_mul_expr()
                # C pointer arithmetic: ptr - n means ptr - n*sizeof(*ptr) bytes.
                lhs_ty = lhs.ty if isinstance(lhs, (Value, Const)) else None
                rhs_ty = rhs.ty if isinstance(rhs, (Value, Const)) else None
                if isinstance(lhs_ty, PtrTy) and not isinstance(rhs_ty, PtrTy):
                    rhs = self._scale_ptr_arith_offset(lhs_ty, rhs)
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
                        # &global_arr[idx] — compute element address
                        if self._match(TokKind.LBRACKET):
                            index = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            elem_size = (cv.ty.pointee.size
                                         if isinstance(cv.ty, PtrTy)
                                         else 4)
                            if elem_size != 1:
                                idx_ty = index.ty if isinstance(index, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, index,
                                                   Const(idx_ty, elem_size)))
                                index = scaled
                            elem_addr = self._new_val("elem_addr", cv.ty)
                            self._emit(BinInst(elem_addr, BinOp.ADD, addr_val, index))
                            # &g_struct_arr[idx].field[.subfield...] — chain field addresses
                            cur_addr2 = elem_addr
                            cur_ty2 = cv.ty.pointee if isinstance(cv.ty, PtrTy) else None
                            while (self._at(TokKind.DOT)
                                   and isinstance(cur_ty2, StructTy)):
                                self._advance()  # consume '.'
                                field = self._expect(TokKind.IDENT).value
                                field_off = cur_ty2.field_offset(field)
                                field_ty = cur_ty2.field_type(field)
                                field_ptr_ty = PtrTy(field_ty, cv.ty.addr_space)
                                field_addr = self._new_val("faddr", field_ptr_ty)
                                self._emit(BinInst(field_addr, BinOp.ADD,
                                                   cur_addr2, Const(INT32, field_off)))
                                cur_addr2 = field_addr
                                cur_ty2 = field_ty
                            return cur_addr2
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
                            # &arr[idx].field[.subfield...] — chain field addresses
                            cur_addr = addr
                            cur_ty = var.ty.pointee
                            while (self._at(TokKind.DOT)
                                   and isinstance(cur_ty, StructTy)):
                                self._advance()  # consume '.'
                                field = self._expect(TokKind.IDENT).value
                                field_off = cur_ty.field_offset(field)
                                field_ty = cur_ty.field_type(field)
                                field_ptr_ty = PtrTy(field_ty, var.ty.addr_space)
                                field_addr = self._new_val("faddr", field_ptr_ty)
                                self._emit(BinInst(field_addr, BinOp.ADD,
                                                   cur_addr, Const(INT32, field_off)))
                                cur_addr = field_addr
                                cur_ty = field_ty
                            return cur_addr  # return address, no load
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
                    elif isinstance(var.ty, StructTy):
                        # If followed by '.' or '->', fall through to the generic
                        # handler so &struct.field is processed correctly.
                        _next_pos_after_ident = self._pos + 1
                        _has_member_access = (
                            _next_pos_after_ident < len(self._toks)
                            and self._toks[_next_pos_after_ident].kind in (
                                TokKind.DOT, TokKind.ARROW))
                        if not _has_member_access:
                            # &struct_var — allocate a .local struct spill slot,
                            # store current field values, return LOCAL pointer.
                            # After the call the spilled_out reload reloads fields.
                            self._advance()  # consume ident
                            spill_name = f"_spill_{name}"
                            local_ty = PtrTy(var.ty, AddrSpace.LOCAL)
                            if spill_name not in self._variables:
                                spill_val = self._new_val(spill_name, local_ty)
                                self._variables[spill_name] = spill_val
                                if not hasattr(self._kernel, '_local_decls'):
                                    self._kernel._local_decls = []
                                self._kernel._local_decls.append(
                                    (spill_name, var.ty, 1, spill_val))
                            else:
                                spill_val = self._variables[spill_name]
                            # Store current field values to their byte offsets
                            for _sfname, _sfty in var.ty.fields:
                                if isinstance(_sfty, (ScalarTy, PtrTy)):
                                    _sfkey = f"{name}_{_sfname}"
                                    _sfval = self._variables.get(_sfkey)
                                    if _sfval is not None:
                                        _sfoff = var.ty.field_offset(_sfname)
                                        _sfaddr = self._new_val(
                                            "faddr", PtrTy(_sfty, AddrSpace.LOCAL))
                                        self._emit(BinInst(
                                            _sfaddr, BinOp.ADD, spill_val,
                                            Const(INT32, _sfoff)))
                                        self._emit(StoreInst(
                                            addr=_sfaddr, value=_sfval))
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
        if self._match(TokKind.PLUS):
            # Unary +: no-op in C — just return the operand unchanged
            return self._parse_unary_expr()
        if self._match(TokKind.MINUS):
            operand = self._parse_unary_expr()
            # Include Const types: -1LL → dest should be INT64, not INT32
            ty = operand.ty if isinstance(operand, (Value, Const)) else INT32
            zero = Const(FLOAT, 0.0) if (isinstance(ty, ScalarTy) and ty.is_float) else Const(ty, 0)
            # Fold -constant immediately so that case labels like `case -1:` never
            # emit a BinInst into potentially-unreachable after_break stubs.
            folded = self._const_fold(BinOp.SUB, zero, operand)
            if folded is not None:
                return folded
            dest = self._new_val("neg", ty)
            self._emit(BinInst(dest, BinOp.SUB, zero, operand))
            return dest
        if self._match(TokKind.TILDE):
            operand = self._parse_unary_expr()
            ty = operand.ty if isinstance(operand, (Value, Const)) else INT32
            # XOR with all-ones of the same width for correct NOT semantics
            all_ones = Const(ty, -1) if isinstance(ty, ScalarTy) else Const(INT32, -1)
            folded = self._const_fold(BinOp.XOR, operand, all_ones)
            if folded is not None:
                return folded
            dest = self._new_val("bnot", ty)
            self._emit(BinInst(dest, BinOp.XOR, operand, all_ones))
            return dest
        if self._match(TokKind.BANG):
            operand = self._parse_unary_expr()
            # Logical NOT: result is BOOL so that 'int x = !expr' triggers CvtInst
            # (BOOL → INT32 → selp.s32), rather than a no-op BinInst ADD of a predicate.
            dest = self._new_val("lnot", ScalarTy(ScalarType.BOOL))
            self._emit(CmpInst(dest, CmpOp.EQ, operand, Const(INT32, 0)))
            return dest
        # Prefix ++i / --i — increment/decrement before use
        if self._match(TokKind.PLUSPLUS):
            _pre_var = self._peek().value if self._at(TokKind.IDENT) else None
            operand = self._parse_unary_expr()
            if isinstance(operand, Value):
                step = (self._scale_ptr_arith_offset(operand.ty, Const(INT32, 1))
                        if isinstance(operand.ty, PtrTy) else Const(operand.ty, 1))
                new_val = self._new_val(f"{operand.name}_preinc", operand.ty)
                self._emit(BinInst(new_val, BinOp.ADD, operand, step))
                # Use _pre_var only if it directly names the operand (not a struct
                # sentinel whose field was resolved).  For ++s.count, _pre_var="s"
                # but operand=_variables["s_count"]; using "s" would clobber the
                # struct sentinel with a scalar Value, corrupting all further
                # s.field accesses.  Check identity: _variables[_pre_var] is operand.
                update_name = (_pre_var if (_pre_var and _pre_var in self._variables
                                            and self._variables[_pre_var] is operand)
                               else operand.name)
                self._variables[update_name] = new_val
                return new_val  # pre-increment returns the new value
            return operand
        if self._match(TokKind.MINUSMINUS):
            _pre_var = self._peek().value if self._at(TokKind.IDENT) else None
            operand = self._parse_unary_expr()
            if isinstance(operand, Value):
                step = (self._scale_ptr_arith_offset(operand.ty, Const(INT32, 1))
                        if isinstance(operand.ty, PtrTy) else Const(operand.ty, 1))
                new_val = self._new_val(f"{operand.name}_predec", operand.ty)
                self._emit(BinInst(new_val, BinOp.SUB, operand, step))
                update_name = (_pre_var if (_pre_var and _pre_var in self._variables
                                            and self._variables[_pre_var] is operand)
                               else operand.name)
                self._variables[update_name] = new_val
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
        # Capture the source variable name before parsing the primary expression.
        # Values resolved from _variables may have a .name that differs from the
        # key used in _variables (e.g. `j = n-1` stores _variables["j"] = Value("compound"))
        # so post-increment/decrement must update _variables["j"], not _variables["compound"].
        _src_var_name = None
        if self._at(TokKind.IDENT):
            cand = self._peek().value
            if cand in self._variables:
                _src_var_name = cand
        lhs = self._parse_primary_expr()

        while True:
            # i++ / i--
            if self._match(TokKind.PLUSPLUS):
                if isinstance(lhs, Value):
                    old = lhs
                    step = (self._scale_ptr_arith_offset(old.ty, Const(INT32, 1))
                            if isinstance(old.ty, PtrTy) else Const(old.ty, 1))
                    new_val = self._new_val(f"{old.name}_inc", old.ty)
                    self._emit(BinInst(new_val, BinOp.ADD, old, step))
                    update_name = _src_var_name if _src_var_name else old.name
                    self._variables[update_name] = new_val
                    lhs = old  # post-increment returns old value
                _src_var_name = None
                continue
            if self._match(TokKind.MINUSMINUS):
                if isinstance(lhs, Value):
                    old = lhs
                    step = (self._scale_ptr_arith_offset(old.ty, Const(INT32, 1))
                            if isinstance(old.ty, PtrTy) else Const(old.ty, 1))
                    new_val = self._new_val(f"{old.name}_dec", old.ty)
                    self._emit(BinInst(new_val, BinOp.SUB, old, step))
                    update_name = _src_var_name if _src_var_name else old.name
                    self._variables[update_name] = new_val
                    lhs = old
                _src_var_name = None
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
                            _src_var_name = key
                        else:
                            # Dynamic index into struct array member: return field_0 as fallback
                            key0 = f"{var_name}_{member}_0"
                            if key0 not in self._variables:
                                fval = self._new_val(key0, elem_ty)
                                self._variables[key0] = fval
                            lhs = self._variables[key0]
                            _src_var_name = key0
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
                        # Update _src_var_name so that postfix ++/-- on a struct
                        # field (s.frame++) writes back to the field key ("s_frame")
                        # rather than the struct variable name ("s"), which would
                        # overwrite the struct sentinel with a scalar Value.
                        _src_var_name = field_key
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
                        addr = self._new_val("faddr", PtrTy(elem_ty, lhs.ty.addr_space))
                        if isinstance(idx_expr, Const):
                            k = int(idx_expr.value) % n
                            field_off = sty.field_offset(f"{member}_{k}")
                            self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        else:
                            # Dynamic index: base + idx * elem_size
                            base_off = sty.field_offset(f"{member}_0")
                            base_addr = self._new_val("base", PtrTy(elem_ty, lhs.ty.addr_space))
                            self._emit(BinInst(base_addr, BinOp.ADD, lhs, Const(INT32, base_off)))
                            elem_size = elem_ty.size
                            idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                            scaled = self._new_val("scale", idx_ty)
                            self._emit(BinInst(scaled, BinOp.MUL, idx_expr,
                                               Const(idx_ty, elem_size)))
                            self._emit(BinInst(addr, BinOp.ADD, base_addr, scaled))
                        if isinstance(elem_ty, StructTy):
                            # Struct element: keep as pointer for chained .field access
                            lhs = addr
                        else:
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
                        addr = self._new_val("faddr", PtrTy(elem_ty, lhs.ty.addr_space))
                        if isinstance(idx_expr, Const):
                            k = int(idx_expr.value) % n
                            field_off = sty.field_offset(f"{member}_{k}")
                            self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                        else:
                            # Dynamic index: base + idx * elem_size
                            base_off = sty.field_offset(f"{member}_0")
                            base_addr = self._new_val("base", PtrTy(elem_ty, lhs.ty.addr_space))
                            self._emit(BinInst(base_addr, BinOp.ADD, lhs, Const(INT32, base_off)))
                            elem_size = elem_ty.size
                            idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                            scaled = self._new_val("scale", idx_ty)
                            self._emit(BinInst(scaled, BinOp.MUL, idx_expr,
                                               Const(idx_ty, elem_size)))
                            self._emit(BinInst(addr, BinOp.ADD, base_addr, scaled))
                        if isinstance(elem_ty, StructTy):
                            # Struct element: keep as pointer for chained .field access
                            lhs = addr
                        else:
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

        if tok.kind == TokKind.IDENT or tok.kind in self._SOFT_KW_AS_IDENT:
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
                    args.append(self._parse_assign_expr())
                    while self._match(TokKind.COMMA):
                        args.append(self._parse_assign_expr())
                self._expect(TokKind.RPAREN)

                if name == '__syncthreads':
                    self._emit(CallInst(None, '__syncthreads', args))
                    return Const(VOID, 0)
                elif name in ('tex1Dfetch', 'tex2D', 'tex3D'):
                    # Texture fetch intrinsics — return float (x component of v4)
                    dest = self._new_val(name, FLOAT)
                    self._emit(CallInst(dest, name, args))
                    return dest
                elif name in ('surf1Dread',):
                    # Surface read — returns int32
                    dest = self._new_val(name, INT32)
                    self._emit(CallInst(dest, name, args))
                    return dest
                elif name in ('surf1Dwrite',):
                    # Surface write — void
                    self._emit(CallInst(None, name, args))
                    return Const(VOID, 0)
                elif name in ('__cp_async_bulk_tensor_1d', '__cp_async_bulk_tensor_2d'):
                    # TMA bulk tensor copy — void
                    self._emit(CallInst(None, name, args))
                    return Const(VOID, 0)
                elif name in ('__cp_async_bulk_commit_group',):
                    # TMA commit group — void, no args
                    self._emit(CallInst(None, name, args))
                    return Const(VOID, 0)
                elif name in ('__cp_async_bulk_wait_group',):
                    # TMA wait group — void, one int arg
                    self._emit(CallInst(None, name, args))
                    return Const(VOID, 0)
                elif name in ('__mbarrier_init',):
                    # mbarrier init — void
                    self._emit(CallInst(None, name, args))
                    return Const(VOID, 0)
                elif name in ('__mbarrier_try_wait_parity',):
                    # mbarrier try_wait_parity — returns int (0 or 1)
                    dest = self._new_val(name, INT32)
                    self._emit(CallInst(dest, name, args))
                    return dest
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
                elif name in ('__ldg', '__ldcg', '__ldcs', '__ldlu', '__ldca'):
                    # Cache-hint loads: __ldg(ptr) → ld.global.nc; others → ld.global
                    if args:
                        ptr_arg = args[0]
                        if isinstance(ptr_arg, Value) and isinstance(ptr_arg.ty, PtrTy):
                            if name == '__ldg':
                                # Re-wrap with CONST addr space for nc load
                                nc_ptr = Value(ptr_arg.name, PtrTy(ptr_arg.ty.pointee, AddrSpace.CONST), ptr_arg.id)
                                dest = self._new_val("ldg", ptr_arg.ty.pointee)
                                self._emit(LoadInst(dest, nc_ptr))
                            else:
                                # Drop cache hint — emit regular global load
                                dest = self._new_val(name[2:], ptr_arg.ty.pointee)
                                self._emit(LoadInst(dest, ptr_arg))
                            return dest
                    return Const(INT32, 0)
                elif name == 'printf':
                    # printf("fmt", args...) — emit PrintfInst
                    # The first arg was STRING_LIT; it set self._last_string_lit
                    fmt_str = getattr(self, '_last_string_lit', '')
                    printf_args = args[1:] if len(args) > 1 else []
                    self._emit(PrintfInst(fmt_str, printf_args))
                    return Const(VOID, 0)
                elif hasattr(self, '_recursive_device_funcs') and name in self._recursive_device_funcs:
                    # Recursive device function — emit a real call (not inlined)
                    dfunc = self._device_funcs[name]
                    ret_ty = dfunc['ret_ty']
                    if isinstance(ret_ty, ScalarTy) and ret_ty.scalar == ScalarType.VOID:
                        dest = None
                    else:
                        dest = self._new_val(f"{name}_ret", ret_ty)
                    self._emit(CallInst(dest, f"__devfn_{name}", args))
                    return dest if dest is not None else Const(VOID, 0)
                elif hasattr(self, '_device_funcs') and name in self._device_funcs:
                    # Inline __device__ function with multi-return support
                    dfunc = self._device_funcs[name]
                    saved_vars = dict(self._variables)
                    saved_pos = self._pos
                    saved_inline_target = self._inline_return_target

                    # Identify caller variables passed by pointer (&x pattern).
                    # After the inline, those variables may have been modified
                    # through their spill slot — we must reload them.
                    spilled_out: list[tuple[str, type, object]] = []
                    for arg in args:
                        if (isinstance(arg, Value)
                                and isinstance(arg.ty, PtrTy)
                                and arg.ty.addr_space == AddrSpace.LOCAL):
                            # Find the original var name by matching spill pointer id
                            for vname, vval in saved_vars.items():
                                if (vname.startswith('_spill_')
                                        and isinstance(vval, Value)
                                        and vval.id == arg.id):
                                    orig_name = vname[len('_spill_'):]
                                    if (orig_name in saved_vars
                                            and isinstance(saved_vars[orig_name], Value)):
                                        spilled_out.append(
                                            (orig_name, saved_vars[orig_name].ty, arg))
                                    break

                    # Bind arguments to parameters.
                    # For struct arguments, member access inside the inline body
                    # uses "{lhs.name}_{field}" as the lookup key, where lhs.name
                    # is the Value's .name attribute on the sentinel.  If we just
                    # store `_variables[pname] = arg` (where arg.name may differ
                    # from pname), then `param.field` resolves to `arg.name_field`
                    # rather than `pname_field`.  Two problems follow:
                    #   1. If the inline body declares a local with arg.name (e.g.
                    #      `Vec3 r;` inside cross() when caller also has `Vec3 r`),
                    #      the field keys get overwritten by the new declaration,
                    #      and the formula reads zero-stubs instead of loaded values.
                    #   2. The inline scope's new field Values collide with the
                    #      caller's field Values sharing the same key.
                    # Fix: for struct params, create a fresh sentinel with name=pname
                    # and pre-bind pname_field → caller's field value so that the
                    # inline body's accesses use isolated, correctly-valued keys.
                    for (pname, pty), arg in zip(dfunc['params'], args):
                        if isinstance(pty, StructTy) and isinstance(arg, Value):
                            # Fresh sentinel so that `param.field` → `pname_field`
                            param_sentinel = self._new_val(pname, pty)
                            self._variables[pname] = param_sentinel
                            if (isinstance(arg.ty, PtrTy)
                                    and isinstance(arg.ty.pointee, StructTy)):
                                # arg is a pointer-to-struct (e.g. array[i] for a
                                # struct array keeps the address rather than loading
                                # the struct).  Load each field (and sub-fields of
                                # nested structs) so the inline body can reference
                                # pname_field and pname_nested_subfield.
                                _asp = arg.ty.addr_space
                                def _load_param_fields(
                                        base_ptr, src_sty, dest_prefix, base_off=0):
                                    for _fname, _fty in src_sty.fields:
                                        _off = src_sty.field_offset(_fname) + base_off
                                        _pkey = f"{dest_prefix}_{_fname}"
                                        if isinstance(_fty, ScalarTy):
                                            _faddr = self._new_val(
                                                "faddr", PtrTy(_fty, _asp))
                                            self._emit(BinInst(
                                                _faddr, BinOp.ADD, base_ptr,
                                                Const(INT32, _off)))
                                            _loaded = self._new_val(_pkey, _fty)
                                            self._emit(LoadInst(_loaded, _faddr))
                                            self._variables[_pkey] = _loaded
                                        elif isinstance(_fty, PtrTy):
                                            # Pointer field: load 64-bit address
                                            _faddr = self._new_val(
                                                "faddr", PtrTy(_fty, _asp))
                                            self._emit(BinInst(
                                                _faddr, BinOp.ADD, base_ptr,
                                                Const(INT32, _off)))
                                            _loaded = self._new_val(_pkey, _fty)
                                            self._emit(LoadInst(_loaded, _faddr))
                                            self._variables[_pkey] = _loaded
                                        elif isinstance(_fty, StructTy):
                                            # Nested struct: create sentinel, recurse
                                            if _pkey not in self._variables:
                                                _nsent = self._new_val(_pkey, _fty)
                                                self._variables[_pkey] = _nsent
                                            _load_param_fields(
                                                base_ptr, _fty, _pkey, _off)
                                _load_param_fields(arg, arg.ty.pointee, pname)
                            else:
                                # Pre-bind per-field keys from caller's field vars
                                for _fname, _fty in pty.fields:
                                    if not isinstance(_fty, (ScalarTy, PtrTy)):
                                        continue
                                    caller_field_key = f"{arg.name}_{_fname}"
                                    param_field_key  = f"{pname}_{_fname}"
                                    if caller_field_key in self._variables:
                                        self._variables[param_field_key] = \
                                            self._variables[caller_field_key]
                        else:
                            self._variables[pname] = arg

                    # Create return destination and merge block
                    ret_ty = dfunc['ret_ty']
                    return_dest = self._new_val(f"{name}_ret", ret_ty)
                    return_merge = self._new_block(f"inline_{name}_merge")
                    self._inline_return_target = (return_dest, return_merge.label)
                    # Pre-allocate per-field return Values for struct-returning
                    # functions so that every return path writes to the SAME
                    # canonical field Values (multiple BinInst defs per Value).
                    # The verifier skips dominance checks for multi-def Values,
                    # exactly mirroring how scalar multi-return already works.
                    if isinstance(ret_ty, StructTy):
                        _pre_ret_fields: dict = {}
                        for _rf, _rft in ret_ty.fields:
                            if isinstance(_rft, ScalarTy):
                                _pre_ret_fields[_rf] = self._new_val(
                                    f"{return_dest.name}_{_rf}", _rft)
                        self._inline_struct_return_fields[return_dest.id] = _pre_ret_fields

                    # Isolate the inlined body from the caller's scope stack.
                    # Without this, _declare_local() calls inside the inlined
                    # body add variable names to the CALLER's current scope set.
                    # At the caller's scope exit, those names would be treated
                    # as shadowing declarations and the outer binding would be
                    # restored — erasing any modifications the caller made to
                    # same-named variables before the call (e.g. an accumulator
                    # named 'result' when abs_val also declares 'result').
                    saved_scope_stack = self._scope_locals_stack
                    self._scope_locals_stack = []

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
                    # Reload caller variables that were passed by pointer.
                    # The inlined function may have modified them through the
                    # spill slot — emit ld.local to pick up any changes.
                    for (orig_name, orig_ty, spill_ptr) in spilled_out:
                        if isinstance(orig_ty, StructTy):
                            # For struct-typed spill slots, reload each scalar
                            # field individually using the field's byte offset.
                            for _fname, _fty in orig_ty.fields:
                                if not isinstance(_fty, (ScalarTy, PtrTy)):
                                    continue
                                _off = orig_ty.field_offset(_fname)
                                _faddr = self._new_val(
                                    "faddr", PtrTy(_fty, AddrSpace.LOCAL))
                                self._emit(BinInst(
                                    _faddr, BinOp.ADD, spill_ptr,
                                    Const(INT32, _off)))
                                _loaded = self._new_val(
                                    f"{orig_name}_{_fname}", _fty)
                                self._emit(LoadInst(_loaded, _faddr))
                                self._variables[f"{orig_name}_{_fname}"] = _loaded
                        else:
                            updated = self._new_val(orig_name, orig_ty)
                            self._emit(LoadInst(updated, spill_ptr))
                            self._variables[orig_name] = updated
                    self._inline_return_target = saved_inline_target
                    self._scope_locals_stack = saved_scope_stack
                    # Expose per-field return values in _variables so that
                    # downstream member accesses (result.x) and chained inline
                    # calls (f(g(x))) can resolve them without lazy-undefined
                    # creation.  Keyed by "return_dest.name_fieldname" so the
                    # normal field-lookup path finds them.
                    if return_dest.id in self._inline_struct_return_fields:
                        for _rfname, _rfval in self._inline_struct_return_fields[
                                return_dest.id].items():
                            self._variables[f"{return_dest.name}_{_rfname}"] = _rfval
                    return return_dest
                else:
                    # Infer return type for known math intrinsics; fallback INT32.
                    _void_stmts   = ('__threadfence', '__threadfence_block',
                                     '__threadfence_system',
                                     '__nanosleep', '__trap', '__brkpt')
                    _float_unary  = ('sqrtf','rsqrtf','rcpf','fabsf','sinf','cosf',
                                     'tanf','expf','exp2f','exp10f','logf','log2f','log10f',
                                     'floorf','ceilf','roundf','truncf',
                                     'atanf','asinf','acosf',
                                     'sqrt','rsqrt','fabs','sin','cos',
                                     'exp','exp2','exp10','log','log2','log10',
                                     'floor','ceil','round','trunc',
                                     'atan','asin','acos',
                                     '__fsqrt_rn','__fsqrt_rd','__fsqrt_ru','__fsqrt_rz',
                                     '__frcp_rn', '__frcp_rd', '__frcp_ru', '__frcp_rz',
                                     '__frsqrt_rn')
                    _float_binary = ('fminf','fmaxf','fmodf','powf',
                                     'fmin','fmax','fmod','pow','hypotf','atan2f')
                    _float_ternary = ('fmaf', 'fma',
                                     '__fmaf_rn', '__fmaf_rd', '__fmaf_ru', '__fmaf_rz',
                                     '__fma_rn', '__fma_rd', '__fma_ru', '__fma_rz')
                    _int_unary    = ('abs',)
                    _int_binary   = ('min','max')
                    _uint_return  = ('__activemask',)
                    _sync_ops     = ('__syncwarp',)
                    _sync_reduce  = ('__syncthreads_count', '__syncthreads_and',
                                     '__syncthreads_or')
                    _int_ops      = ('__popc', '__popcll', '__clz', '__clzll',
                                     '__brev', '__brevll', '__ffs', '__ffsll')
                    _clock_ops    = ('clock64', '__clock64', 'clock')
                    _pred_int     = ('isnan', 'isinf', 'isfinite',
                                     '__isnan', '__isinf', '__isfinite',
                                     '__isnanf', '__isinff', '__isfinitef')
                    _float2int    = ('__float2int_rn', '__float2int_rd',
                                     '__float2int_ru', '__float2int_rz',
                                     '__float2uint_rn', '__float2uint_rd',
                                     '__float2uint_ru', '__float2uint_rz',
                                     '__float2ll_rn', '__float2ll_rd',
                                     '__float2ll_ru', '__float2ll_rz',
                                     '__float2ull_rn', '__float2ull_rd',
                                     '__float2ull_ru', '__float2ull_rz',
                                     '__double2int_rn', '__double2int_rz',
                                     '__double2uint_rn', '__double2uint_rz',
                                     '__double2ll_rn', '__double2ll_rz',
                                     '__double2ull_rn', '__double2ull_rz',
                                     '__float_as_int', '__float_as_uint',
                                     '__double_as_longlong', '__double_as_ulonglong')
                    _int2float    = ('__int2float_rn', '__int2float_rd',
                                     '__int2float_ru', '__int2float_rz',
                                     '__uint2float_rn', '__uint2float_rd',
                                     '__uint2float_ru', '__uint2float_rz',
                                     '__ll2float_rn', '__ll2float_rz',
                                     '__ull2float_rn', '__ull2float_rz',
                                     '__int2double_rn', '__ll2double_rn',
                                     '__uint2double_rn', '__ull2double_rn',
                                     '__int_as_float', '__uint_as_float',
                                     '__longlong_as_double', '__ulonglong_as_double')
                    _sad_ops      = ('__sad', '__usad')
                    _dp4a_ops     = ('__dp4a', '__dp4a_u', '__dp4a_su', '__dp4a_us')
                    _warp_reduce_int = ('__reduce_add_sync', '__reduce_min_sync',
                                        '__reduce_max_sync')
                    _warp_reduce_uint = ('__reduce_and_sync', '__reduce_or_sync',
                                         '__reduce_xor_sync', '__reduce_umin_sync',
                                         '__reduce_umax_sync')
                    _warp_match   = ('__match_any_sync', '__match_all_sync')
                    _int_binary2  = ('__mul24', '__mulhi')
                    _uint_binary2 = ('__umul24', '__umulhi', '__rhadd',
                                     '__byte_perm',
                                     '__funnelshift_l', '__funnelshift_r',
                                     '__funnelshift_lc', '__funnelshift_rc')
                    # Half-precision (__half) intrinsics
                    _half_unary   = ('__habs', '__hneg', '__hsqrt', '__hrcp',
                                     '__hrsqrt', '__hexp', '__hlog',
                                     '__hceil', '__hfloor', '__hrint', '__htrunc',
                                     '__hcos', '__hsin', '__hexp2', '__hlog2', '__hlog10',
                                     '__half2float', '__low2float', '__high2float')
                    _half_binary  = ('__hmul', '__hmul_rn', '__hmul_sat',
                                     '__hadd_rn', '__hsub', '__hsub_rn',
                                     '__hdiv', '__hfmin', '__hfmax',
                                     '__hmin', '__hmax')
                    _half_ternary = ('__hfma', '__hfma_sat', '__hfma_relu')
                    _half_cmp_int = ('__hgt', '__hlt', '__hge', '__hle',
                                     '__heq', '__hne', '__hisnan', '__hisinf')
                    _half_cvt     = ('__float2half', '__float2half_rn',
                                     '__float2half_rd', '__float2half_ru',
                                     '__float2half_rz',
                                     '__int2half_rn', '__int2half_rd',
                                     '__int2half_ru', '__int2half_rz',
                                     '__uint2half_rn', '__uint2half_rd',
                                     '__uint2half_ru', '__uint2half_rz',
                                     '__short2half_rn', '__ushort2half_rn',
                                     '__ll2half_rn', '__ll2half_rz',
                                     '__ull2half_rn', '__ull2half_rz',
                                     '__ushort_as_half', '__short_as_half')
                    _half_to_bits = ('__half_as_ushort', '__half_as_short')
                    _half2int_cvt = ('__half2int_rn', '__half2int_rz',
                                     '__half2uint_rn', '__half2uint_rz',
                                     '__half2short_rn', '__half2short_rz',
                                     '__half2ushort_rn', '__half2ushort_rz',
                                     '__half2ll_rn', '__half2ll_rz',
                                     '__half2ull_rn', '__half2ull_rz')
                    _hadd_fp16    = ('__hadd', '__hadd_sat')  # overloaded: half add when HALF, else int halving
                    if name in _void_stmts:
                        self._emit(CallInst(None, name, args))
                        return Const(VOID, 0)
                    elif name in _sync_ops:
                        self._emit(CallInst(None, name, args))
                        return Const(VOID, 0)
                    elif name in _sync_reduce:
                        dest = self._new_val(name, INT32)
                        self._emit(CallInst(dest, name, args))
                        return dest
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
                    elif name in _clock_ops:
                        # clock64() → u64; clock() → u32
                        ret_ty = UINT64 if '64' in name else UINT32
                        dest = self._new_val(name, ret_ty)
                        self._emit(CallInst(dest, name, args))
                        return dest
                    elif name in _pred_int:
                        ret_ty = INT32
                    elif name in _float2int:
                        ret_ty = INT64 if ('ll' in name or 'longlong' in name) else INT32
                    elif name in _int2float:
                        ret_ty = DOUBLE if 'double' in name else FLOAT
                    elif name in _sad_ops:
                        ret_ty = UINT32 if name == '__usad' else INT32
                    elif name in _dp4a_ops:
                        ret_ty = INT32
                    elif name in _warp_reduce_int:
                        ret_ty = INT32
                    elif name in _warp_reduce_uint:
                        ret_ty = UINT32
                    elif name in _warp_match:
                        ret_ty = UINT32
                    elif name in _int_binary2:
                        ret_ty = INT32
                    elif name in _uint_binary2:
                        ret_ty = UINT32
                    elif name in _half_unary:
                        ret_ty = HALF if name != '__half2float' and 'float' not in name else FLOAT
                    elif name in _half_binary:
                        ret_ty = HALF
                    elif name in _half_ternary:
                        ret_ty = HALF
                    elif name in _half_cmp_int:
                        ret_ty = INT32
                    elif name in _half_cvt:
                        ret_ty = HALF
                    elif name in _half_to_bits:
                        ret_ty = UINT16
                    elif name in _half2int_cvt:
                        # half → integer: width/sign determined by function name
                        if 'll' in name or 'ull' in name:
                            ret_ty = UINT64 if 'ull' in name else INT64
                        elif 'uint' in name or 'ushort' in name:
                            ret_ty = UINT32
                        else:
                            ret_ty = INT32
                    elif name in _hadd_fp16:
                        # __hadd: half add when arg is HALF, else integer halving add
                        a_ty = args[0].ty if args and isinstance(args[0], Value) else INT32
                        if isinstance(a_ty, ScalarTy) and a_ty.scalar == ScalarType.HALF:
                            ret_ty = HALF
                        else:
                            ret_ty = INT32
                    elif name in _float_unary:
                        # Preserve double precision: sin(double) → double, sinf(float) → float
                        _a0 = args[0] if args else None
                        _a0_ty = _a0.ty if isinstance(_a0, (Value, Const)) else FLOAT
                        ret_ty = DOUBLE if (isinstance(_a0_ty, ScalarTy) and _a0_ty.scalar == ScalarType.DOUBLE) else FLOAT
                    elif name in _float_binary:
                        _a0 = args[0] if args else None
                        _a0_ty = _a0.ty if isinstance(_a0, (Value, Const)) else FLOAT
                        ret_ty = DOUBLE if (isinstance(_a0_ty, ScalarTy) and _a0_ty.scalar == ScalarType.DOUBLE) else FLOAT
                    elif name in _float_ternary:
                        _a0 = args[0] if args else None
                        _a0_ty = _a0.ty if isinstance(_a0, (Value, Const)) else FLOAT
                        ret_ty = DOUBLE if (isinstance(_a0_ty, ScalarTy) and _a0_ty.scalar == ScalarType.DOUBLE) else FLOAT
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
                var = self._variables[name]
                # Auto-dereference __shared__ scalar vars used as rvalues.
                # total declared as __shared__ float total; has PtrTy(FLOAT, SHARED).
                # When accessed without '[' or '.', emit a load to return the scalar value.
                # DOT suppressed: stree.field must go through the ptr-to-struct path in
                # _parse_postfix_expr so that field address computation is emitted correctly.
                if (name in self._shared_scalars
                        and isinstance(var.ty, PtrTy)
                        and not self._at(TokKind.LBRACKET)
                        and not self._at(TokKind.DOT)):
                    loaded = self._new_val(name, var.ty.pointee)
                    self._emit(LoadInst(loaded, var))
                    return loaded
                return var

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

            # warpSize: CUDA built-in constant = 32 (PTX %warpsize register)
            if name == 'warpSize':
                dest = self._new_val('warpSize', UINT32)
                self._emit(CallInst(dest, 'warpSize', []))
                return dest

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
            # Detect C99 compound literal `(TYPE){fields}` or a C-style
            # cast `(TYPE)expr`.  FORGE_AGG expands to the compound-literal
            # form.  The lookahead: an opening LPAREN followed by a token
            # that begins a type (KW_INT, KW_FLOAT, ..., or a typedef'd /
            # struct identifier).  Save self._pos so we can roll back if
            # what follows turns out not to be a type after all.
            _type_start_kinds = (
                TokKind.KW_VOID, TokKind.KW_INT, TokKind.KW_FLOAT,
                TokKind.KW_DOUBLE, TokKind.KW_HALF, TokKind.KW_CHAR,
                TokKind.KW_BOOL, TokKind.KW_SHORT, TokKind.KW_LONG,
                TokKind.KW_SIGNED, TokKind.KW_UNSIGNED,
                TokKind.KW_STRUCT, TokKind.KW_UNION, TokKind.KW_ENUM,
                TokKind.KW_CONST, TokKind.KW_VOLATILE,
            )
            _is_type_lookahead = (
                self._peek().kind in _type_start_kinds
                or (self._at(TokKind.IDENT)
                    and (self._peek().value in self._typedefs
                         or self._peek().value in self._struct_types))
            )
            if _is_type_lookahead:
                _saved = self._pos
                try:
                    cast_ty = self._parse_type_with_ptr()
                    if self._at(TokKind.RPAREN):
                        self._advance()  # consume ')'
                        if self._at(TokKind.LBRACE) and isinstance(cast_ty, StructTy):
                            # C99 compound literal: (TYPE){f0, f1, ...}
                            return self._parse_compound_literal(cast_ty)
                        # C-style cast: (TYPE)expr
                        operand = self._parse_unary_expr()
                        dest = self._new_val("cast", cast_ty)
                        self._emit(CvtInst(dest, operand))
                        return dest
                except Exception:
                    self._pos = _saved
            expr = self._parse_expr()
            self._expect(TokKind.RPAREN)
            return expr

        raise ParseError(f"Line {tok.line}: unexpected token '{tok.value}'")

    def _parse_compound_literal(self, sty: 'StructTy') -> 'Value':
        """Parse a C99 compound literal `{f0, f1, ...}` after the
        opening `(TYPE)` has been consumed.  Returns a Value of struct
        type with per-field Values registered in
        self._inline_struct_return_fields[dest.id] so that the existing
        return-statement and field-access machinery can pick them up.
        """
        self._expect(TokKind.LBRACE)
        field_vals: list = []
        if not self._at(TokKind.RBRACE):
            field_vals.append(self._parse_assign_expr())
            while self._match(TokKind.COMMA):
                if self._at(TokKind.RBRACE):
                    break  # trailing comma
                field_vals.append(self._parse_assign_expr())
        self._expect(TokKind.RBRACE)
        # Construct a struct Value and populate its per-field map.  Mirror
        # the protocol used for inline struct returns: a dict keyed by the
        # destination Value's id, mapping field name → field Value.
        dest = self._new_val("compound", sty)
        fmap: dict = {}
        for (fname, fty), val in zip(sty.fields, field_vals):
            if not isinstance(fty, ScalarTy):
                continue
            fval = self._new_val(f"{dest.name}_{fname}", fty)
            zero = Const(fty, 0.0 if fty.is_float else 0)
            self._emit(BinInst(fval, BinOp.ADD, val, zero))
            fmap[fname] = fval
            # Also expose under the compound name so dot-access can
            # find the field via _variables lookup.
            self._variables[f"{dest.name}_{fname}"] = fval
        self._inline_struct_return_fields[dest.id] = fmap
        return dest

    # -- Statement parsing ---------------------------------------------------

    def _parse_stmt(self):
        tok = self._peek()

        # const/volatile/static/inline/register declaration: skip the qualifier and parse as normal
        _ignorable_quals = (TokKind.KW_CONST, TokKind.KW_VOLATILE, TokKind.KW_STATIC)
        _stmt_is_volatile = False
        while tok.kind in _ignorable_quals:
            if tok.kind == TokKind.KW_VOLATILE:
                _stmt_is_volatile = True
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

        # Inline assembly: asm [volatile] ("template" : "=r"(out) : "r"(in))
        if tok.kind == TokKind.IDENT and tok.value in ('asm', '__asm__', '__asm'):
            from ..ir.nodes import AsmInst
            self._advance()  # consume 'asm'
            if self._at(TokKind.KW_VOLATILE) or (self._at(TokKind.IDENT) and self._peek().value == 'volatile'):
                self._advance()
            self._expect(TokKind.LPAREN)
            tmpl = ''
            if self._at(TokKind.STRING_LIT):
                tmpl = self._peek().value[1:-1]  # strip quotes
                self._advance()
            outputs = []
            inputs = []
            if self._match(TokKind.COLON):
                # Output operands: "=r"(var), ...
                while self._at(TokKind.STRING_LIT):
                    constraint = self._peek().value[1:-1]
                    self._advance()
                    self._expect(TokKind.LPAREN)
                    var_name = self._expect(TokKind.IDENT).value
                    self._expect(TokKind.RPAREN)
                    old_val = self._variables.get(var_name)
                    ty = old_val.ty if old_val else INT32
                    # Create fresh Value — the asm instruction defines this
                    new_val = self._new_val(var_name, ty)
                    outputs.append((constraint, new_val))
                    self._variables[var_name] = new_val
                    if not self._match(TokKind.COMMA):
                        break
                if self._match(TokKind.COLON):
                    # Input operands: "r"(expr), ...
                    while self._at(TokKind.STRING_LIT):
                        constraint = self._peek().value[1:-1]
                        self._advance()
                        self._expect(TokKind.LPAREN)
                        var_expr = self._parse_assign_expr()
                        self._expect(TokKind.RPAREN)
                        inputs.append((constraint, var_expr))
                        if not self._match(TokKind.COMMA):
                            break
                    # Optional clobber list
                    if self._match(TokKind.COLON):
                        while self._at(TokKind.STRING_LIT):
                            self._advance()
                            if not self._match(TokKind.COMMA):
                                break
            self._expect(TokKind.RPAREN)
            self._expect(TokKind.SEMI)
            if tmpl:
                self._emit(AsmInst(tmpl, outputs, inputs))
            return

        # __shared__ declaration: __shared__ type name[size], name[d0][d1]...,
        # or extern __shared__ type name[] (dynamic shared memory, size=0 sentinel).
        if tok.kind == TokKind.KW_SHARED:
            self._advance()
            ty = self._parse_type()
            name = self._expect(TokKind.IDENT).value
            # __shared__ scalar: __shared__ float total; — treat as size-1 array
            # (allocated in .shared, accessed as ptr[0]).
            # Also handles comma-separated: __shared__ int a, b;
            def _declare_shared_scalar(n):
                smem_ty = PtrTy(ty, AddrSpace.SHARED, volatile=_stmt_is_volatile)
                v = self._new_val(n, smem_ty)
                self._variables[n] = v
                self._shared_scalars.add(n)
                if not hasattr(self._kernel, '_shared_decls'):
                    self._kernel._shared_decls = []
                self._kernel._shared_decls.append((n, ty, 1))

            if self._match(TokKind.SEMI):
                _declare_shared_scalar(name)
                return
            if self._at(TokKind.COMMA):
                _declare_shared_scalar(name)
                while self._match(TokKind.COMMA):
                    extra = self._expect(TokKind.IDENT).value
                    _declare_shared_scalar(extra)
                self._expect(TokKind.SEMI)
                return
            self._expect(TokKind.LBRACKET)
            # extern __shared__ float sdata[]; — empty brackets → dynamic
            if self._at(TokKind.RBRACKET):
                self._advance()
                self._expect(TokKind.SEMI)
                smem_ty = PtrTy(ScalarTy(ScalarType.FLOAT) if ty == FLOAT else ty, AddrSpace.SHARED,
                                volatile=_stmt_is_volatile)
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
            # Create a shared-memory pointer variable (inner helper for reuse below)
            def _register_shared_array(arr_name, arr_ty, arr_size, arr_inner_dims):
                if arr_inner_dims:
                    elem_size = arr_ty.size if isinstance(arr_ty, ScalarTy) else 8
                    inner_prod = 1
                    for d in arr_inner_dims:
                        inner_prod *= d
                    self._array_row_strides[arr_name] = inner_prod * elem_size
                smem_ty = PtrTy(ScalarTy(ScalarType.FLOAT) if arr_ty == FLOAT else arr_ty, AddrSpace.SHARED,
                                volatile=_stmt_is_volatile)
                v2 = self._new_val(arr_name, smem_ty)
                self._variables[arr_name] = v2
                self._declare_local(arr_name)
                if not hasattr(self._kernel, '_shared_decls'):
                    self._kernel._shared_decls = []
                self._kernel._shared_decls.append((arr_name, arr_ty, arr_size))

            _register_shared_array(name, ty, size, inner_dims)

            # Handle comma-separated shared array declarations:
            # __shared__ float a[256], b[256];
            while self._match(TokKind.COMMA):
                extra_name = self._expect(TokKind.IDENT).value
                self._expect(TokKind.LBRACKET)
                extra_size_op = self._parse_assign_expr()
                extra_size = int(extra_size_op.value) if isinstance(extra_size_op, Const) else 1
                self._expect(TokKind.RBRACKET)
                extra_inner = []
                while self._at(TokKind.LBRACKET):
                    self._advance()
                    dim_op = self._parse_assign_expr()
                    dim = int(dim_op.value) if isinstance(dim_op, Const) else 1
                    extra_inner.append(dim)
                    extra_size *= dim
                    self._expect(TokKind.RBRACKET)
                _register_shared_array(extra_name, ty, extra_size, extra_inner)

            self._expect(TokKind.SEMI)
            return

        # Variable declaration: type name [= expr] [, name2 [= expr2]] ...;
        # Handles both single and multiple comma-separated declarators.
        # Also handles typedef'd types (e.g. float3, int2, user typedefs).
        if (tok.kind in (TokKind.KW_INT, TokKind.KW_UNSIGNED, TokKind.KW_SIGNED,
                         TokKind.KW_FLOAT, TokKind.KW_DOUBLE, TokKind.KW_VOID,
                         TokKind.KW_LONG, TokKind.KW_HALF, TokKind.KW_CHAR,
                         TokKind.KW_SHORT, TokKind.KW_BOOL,
                         TokKind.KW_STRUCT, TokKind.KW_UNION, TokKind.KW_ENUM)
                or (tok.kind == TokKind.IDENT and (tok.value in self._typedefs
                                                    or tok.value in self._struct_types))):
            ty = self._parse_type_with_ptr()
            while True:
                # Each declarator may have its own pointer stars: int *a, b, *c;
                decl_ty = ty
                while self._match(TokKind.STAR):
                    decl_ty = PtrTy(decl_ty, AddrSpace.GLOBAL)
                name = self._expect_ident().value
                # Skip __attribute__((unused)) or similar GCC qualifiers on local vars
                while self._at(TokKind.IDENT) and self._peek().value == '__attribute__':
                    self._advance()
                    if self._at(TokKind.LPAREN):
                        depth = 1
                        self._advance()
                        while depth > 0 and not self._at(TokKind.EOF):
                            if self._peek().kind == TokKind.LPAREN: depth += 1
                            elif self._peek().kind == TokKind.RPAREN: depth -= 1
                            self._advance()

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
                        self._declare_local(name)
                        if not hasattr(self._kernel, '_local_decls'):
                            self._kernel._local_decls = []
                        self._kernel._local_decls.append((name, decl_ty, count, arr_val))
                        if not self._match(TokKind.COMMA):
                            break
                        continue
                    sentinel = self._new_val(name, decl_ty)
                    self._variables[name] = sentinel
                    self._declare_local(name)
                    def _zero_init_decl_fields(prefix: str, sty: 'StructTy') -> None:
                        for fname, fty in sty.fields:
                            _key = f"{prefix}_{fname}"
                            fval = self._new_val(_key, fty)
                            self._variables[_key] = fval
                            if isinstance(fty, ScalarTy):
                                _fzero = Const(fty, 0.0 if fty.is_float else 0)
                                self._emit(BinInst(fval, BinOp.ADD, _fzero, _fzero))
                            elif isinstance(fty, StructTy):
                                _zero_init_decl_fields(_key, fty)
                    _zero_init_decl_fields(name, decl_ty)
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
                                addr_space = rhs.ty.addr_space
                                # Load all scalar sub-fields recursively (handles
                                # nested structs like Particle.pos.x).
                                def _load_struct_fields(base_ptr, base_sty,
                                                        var_prefix, abs_off=0):
                                    for _fn, _ft in base_sty.fields:
                                        _off = base_sty.field_offset(_fn) + abs_off
                                        if isinstance(_ft, ScalarTy):
                                            _fa = self._new_val(
                                                "faddr", PtrTy(_ft, addr_space))
                                            self._emit(BinInst(
                                                _fa, BinOp.ADD, base_ptr,
                                                Const(INT32, _off)))
                                            _lv = self._new_val(
                                                f"{var_prefix}_{_fn}", _ft)
                                            self._emit(LoadInst(_lv, _fa))
                                            self._variables[f"{var_prefix}_{_fn}"] = _lv
                                        elif isinstance(_ft, StructTy):
                                            # Nested struct sentinel
                                            _sfkey = f"{var_prefix}_{_fn}"
                                            if _sfkey not in self._variables:
                                                _sf_sent = self._new_val(
                                                    _sfkey, _ft)
                                                self._variables[_sfkey] = _sf_sent
                                            _load_struct_fields(
                                                base_ptr, _ft, _sfkey, _off)
                                _load_struct_fields(rhs, sty, name)
                            elif isinstance(rhs, Value) and isinstance(rhs.ty, StructTy):
                                # Direct struct value — from inlined __device__ function return
                                # or struct-to-struct copy.  Look up per-field values.
                                field_map = self._inline_struct_return_fields.get(rhs.id, {})
                                for fname, fty in decl_ty.fields:
                                    if not isinstance(fty, ScalarTy):
                                        continue
                                    src = field_map.get(fname)
                                    if src is None:
                                        # Fall back: same-name source variable in scope
                                        src = self._variables.get(f"{rhs.name}_{fname}")
                                    if src is not None:
                                        fval2 = self._new_val(f"{name}_{fname}", fty)
                                        self._emit(BinInst(
                                            fval2, BinOp.ADD, src,
                                            Const(fty, 0.0 if fty.is_float else 0)))
                                        self._variables[f"{name}_{fname}"] = fval2
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
                    self._declare_local(name)
                    # Store local array info for codegen
                    if not hasattr(self._kernel, '_local_decls'):
                        self._kernel._local_decls = []
                    self._kernel._local_decls.append((name, decl_ty, size, val))
                    # Optional aggregate initializer: int arr[N] = {e0, e1, ...};
                    # Supports nested braces for multi-dim arrays (flattened):
                    #   int m[4][4] = {{1,2,3,4},{5,6,7,8},...}
                    if self._match(TokKind.ASSIGN):
                        if self._at(TokKind.LBRACE):
                            self._advance()  # consume '{'
                            elem_sz = decl_ty.size if isinstance(decl_ty, ScalarTy) else 8
                            init_idx = 0
                            def _emit_flat_elems(depth=0):
                                nonlocal init_idx
                                while not self._at(TokKind.RBRACE) and not self._at(TokKind.EOF):
                                    if self._at(TokKind.LBRACE):
                                        # nested row brace — recurse to flatten
                                        self._advance()
                                        _emit_flat_elems(depth + 1)
                                        self._expect(TokKind.RBRACE)
                                    else:
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
                            _emit_flat_elems()
                            self._expect(TokKind.RBRACE)
                    if not self._match(TokKind.COMMA):
                        break
                    continue

                val = self._new_val(name, decl_ty)
                self._variables[name] = val
                self._declare_local(name)

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
                            # Always emit an explicit copy so 'name' gets its own
                            # canonical register, distinct from 'rhs'.  Aliasing
                            # (self._variables[name] = rhs) breaks nested loops:
                            # `int j = i` would share i's writeback register, so
                            # the inner-loop writeback for j clobbers i's value,
                            # causing the outer loop to exit after one iteration.
                            self._emit(BinInst(val, BinOp.ADD, rhs, Const(decl_ty, 0)))
                            self._variables[name] = val
                else:
                    # Uninitialized declaration: emit a zero-init so `val` has a
                    # defining instruction.  Without this, any use of val before
                    # assignment (e.g. the initial spill-store when &val is taken)
                    # would be flagged as "use of undefined value" by the verifier.
                    # C semantics allow undefined initial values; the zero here is
                    # dead if the variable is assigned before use, and correct if
                    # only written through a pointer (e.g. float lo; set(&lo, ...)).
                    _zero = Const(decl_ty, 0.0 if isinstance(decl_ty, ScalarTy) and decl_ty.is_float else 0)
                    self._emit(BinInst(val, BinOp.ADD, _zero, _zero))

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

            # Push a scope for for-loop init variables (e.g. `int i = 0`).
            # This ensures that after the loop, any outer variable with the
            # same name is restored — matching C's rule that for-loop init
            # declarations are scoped to the loop statement itself.
            for_outer_bindings = dict(self._variables)
            self._scope_locals_stack.append(set())

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
            # Use _cur_block (not cond_bb) so that if the condition contained an
            # inlined function call, the CondBrTerm lands on the inline's merge
            # block (which holds the condition result), not on cond_bb itself.
            # cond_bb still has BrTerm(inline_body) from the inline; the loop
            # back-edge jumps to cond_bb.label which then flows through the inline.
            self._cur_block.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            # Emit body (with break → exit_bb, continue → inc_bb)
            self._pos = body_resume
            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(loop_vars)  # for writeback on break
            self._continue_targets.append(inc_bb.label)
            # Pass loop_vars so the continue handler writes back any variables
            # modified before the continue INTO the live block (not the dead
            # after_continue stub).  Without this, dead_block_elim removes the
            # writeback instruction and identity_fold folds the canonical value
            # to Const(0), silently zeroing variables like neg_count.
            # inc_bb still does its own writeback for the increment expression;
            # that write-back is idempotent (cur_val == init_val → skipped).
            self._continue_snapshots.append(loop_vars)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()
            self._continue_snapshots.pop()
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
                if not isinstance(init_val, Value):
                    continue
                if isinstance(init_val.ty, StructTy):
                    # Restore canonical sentinel pointer (no BinInst needed).
                    if self._variables.get(var_name) is not init_val:
                        self._variables[var_name] = init_val
                    continue
                cur_val = self._variables.get(var_name)
                if cur_val is None or cur_val is init_val:
                    continue
                if isinstance(cur_val, Value) and isinstance(cur_val.ty, StructTy):
                    continue  # current value is also a struct sentinel — skip
                if isinstance(cur_val, Value):
                    self._emit(BinInst(init_val, BinOp.ADD, cur_val, Const(init_val.ty, 0)))
                    self._variables[var_name] = init_val

            # Use _cur_block (not inc_bb) so that if the increment expression
            # contained a ternary, the BrTerm lands on the ternary's merge block.
            self._cur_block.terminator = BrTerm(cond_bb.label)
            self._cur_block = exit_bb

            # Pop the for-loop init scope: restore outer bindings for any
            # variable re-declared in the init (e.g. `int i` when outer `i`
            # exists), and remove purely inner variables.
            for_inner_decls = self._scope_locals_stack.pop()
            for name in for_inner_decls:
                if name in for_outer_bindings:
                    self._variables[name] = for_outer_bindings[name]
                elif name in self._variables:
                    del self._variables[name]
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
            self._cur_block.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._break_snapshots.append(while_entry_vars)  # for writeback on break
            self._continue_targets.append(cond_bb.label)
            self._continue_snapshots.append(while_entry_vars)  # for writeback on continue
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()
            self._continue_snapshots.pop()

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
            self._break_snapshots.append(do_entry_vars)  # for writeback on break
            self._continue_targets.append(cond_bb.label)
            self._continue_snapshots.append(do_entry_vars)  # for writeback on continue
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._break_snapshots.pop()
            self._continue_targets.pop()
            self._continue_snapshots.pop()

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
            self._cur_block.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

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
            # Write back any variables modified before the continue so the
            # continue target (cond_bb for while/do-while) sees updated values.
            if self._continue_snapshots and self._continue_snapshots[-1] is not None:
                self._loop_writeback(self._continue_snapshots[-1])
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
                    if (isinstance(return_dest.ty, StructTy)
                            and isinstance(ret_val, Value)
                            and isinstance(ret_val.ty, StructTy)):
                        # Struct return: emit copies to pre-allocated canonical
                        # field Values.  All return paths write to the same field
                        # Values (multiple BinInst definitions), so the verifier
                        # skips dominance checking — the same mechanism that makes
                        # scalar multi-return correct (same return_dest, multi-def).
                        _pre_ret = self._inline_struct_return_fields.get(
                            return_dest.id, {})
                        _src_fmap = self._inline_struct_return_fields.get(
                            ret_val.id, {})
                        for fname, fty in return_dest.ty.fields:
                            if not isinstance(fty, ScalarTy):
                                continue
                            src = _src_fmap.get(fname)
                            if src is None:
                                src = self._variables.get(
                                    f"{ret_val.name}_{fname}")
                            if src is not None:
                                dest_f = _pre_ret.get(fname)
                                if dest_f is None:
                                    dest_f = self._new_val(
                                        f"{return_dest.name}_{fname}", fty)
                                    _pre_ret[fname] = dest_f
                                self._emit(BinInst(
                                    dest_f, BinOp.ADD, src,
                                    Const(fty, 0.0 if fty.is_float else 0)))
                        if _pre_ret:
                            self._inline_struct_return_fields[return_dest.id] = _pre_ret
                    else:
                        zero = Const(return_dest.ty, 0.0 if (isinstance(return_dest.ty, ScalarTy) and return_dest.ty.is_float) else 0)
                        self._emit(BinInst(return_dest, BinOp.ADD, ret_val, zero))
                self._cur_block.terminator = BrTerm(return_merge_label)
                # Create dead block for any code after this return
                self._cur_block = self._new_block("after_inline_return")
            else:
                self._cur_block.terminator = RetTerm(ret_val=ret_val)
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
                            and (self._toks[assign_pos].kind in _assign_ops
                                 or self._toks[assign_pos].kind in (
                                     TokKind.PLUSPLUS, TokKind.MINUSMINUS))):
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
                # Exception: __shared__ scalars always use StoreInst.
                if _stmt_lhs_name and _stmt_lhs_name in self._shared_scalars:
                    _stmt_lhs_name = None
                update_name = _stmt_lhs_name or lhs.name
                if _stmt_lhs_name and update_name in self._variables:
                    # Pointer variable reassignment: p = new_ptr_value
                    self._variables[update_name] = rhs
                else:
                    # Memory store through pointer: ptr[i] = val or *ptr = val
                    # Struct store: ptr_to_struct = struct_val → expand to per-field stores
                    if (isinstance(rhs, Value) and isinstance(rhs.ty, StructTy)
                            and isinstance(lhs.ty.pointee, StructTy)):
                        _sty = lhs.ty.pointee
                        _fmap = self._inline_struct_return_fields.get(rhs.id, {})
                        for _sf, _sft in _sty.fields:
                            if not isinstance(_sft, ScalarTy):
                                continue
                            _fval = _fmap.get(_sf)
                            if _fval is None:
                                _fval = self._variables.get(f"{rhs.name}_{_sf}")
                            if _fval is None:
                                continue
                            _off = _sty.field_offset(_sf)
                            _faddr = self._new_val("faddr", PtrTy(_sft, lhs.ty.addr_space))
                            self._emit(BinInst(_faddr, BinOp.ADD, lhs, Const(INT32, _off)))
                            self._emit(StoreInst(addr=_faddr, value=_fval))
                    else:
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
                    prev = self._variables[update_name]
                    # For scalar assignments where rhs is a Value with a different
                    # name, emit an identity copy to decouple the target register
                    # from the source.  Without this, loop-carried writeback would
                    # use the source register as the canonical entry Value and
                    # incorrectly modify it instead of the target (e.g. s.count = n
                    # followed by s.count-- would decrement the n param register).
                    if (isinstance(rhs, Value)
                            and isinstance(prev, Value)
                            and isinstance(prev.ty, ScalarTy)
                            and isinstance(rhs.ty, ScalarTy)
                            and rhs.name != update_name):
                        mat = self._new_val(update_name, prev.ty)
                        self._emit(BinInst(mat, BinOp.ADD, rhs,
                                           Const(prev.ty, 0.0 if prev.ty.is_float else 0)))
                        rhs = mat
                    self._variables[update_name] = rhs
                    # For struct-to-struct assignment, propagate per-field values
                    # so downstream field accesses (v.x, v.y, ...) see the new values.
                    if isinstance(rhs, Value) and isinstance(rhs.ty, StructTy):
                        field_map = self._inline_struct_return_fields.get(rhs.id, {})
                        for _fname, _fty in rhs.ty.fields:
                            if not isinstance(_fty, ScalarTy):
                                continue
                            _src_f = field_map.get(_fname)
                            if _src_f is None:
                                _src_f = self._variables.get(f"{rhs.name}_{_fname}")
                            _dest_fkey = f"{update_name}_{_fname}"
                            if _src_f is not None and _dest_fkey in self._variables:
                                _new_fv = self._new_val(_dest_fkey, _fty)
                                self._emit(BinInst(
                                    _new_fv, BinOp.ADD, _src_f,
                                    Const(_fty, 0.0 if _fty.is_float else 0)))
                                self._variables[_dest_fkey] = _new_fv
                        # Restore the struct sentinel with the LHS variable name so
                        # subsequent "s.field" lookups build field_key = "s_field"
                        # (not "rhs_name_field" when rhs came from an inline return
                        # with a different sentinel name, e.g. "add_sample_ret").
                        cur_sent = self._variables.get(update_name)
                        if (isinstance(cur_sent, Value)
                                and isinstance(cur_sent.ty, StructTy)
                                and cur_sent.name != update_name):
                            fresh_sent = self._new_val(update_name, cur_sent.ty)
                            self._variables[update_name] = fresh_sent
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
                    # Distinguish pointer VARIABLE compound (p += n → advance pointer)
                    # from memory compound (*p += n → load, modify, store).
                    # _stmt_lhs_name is set when the statement started with a tracked IDENT.
                    is_ptr_var = (bool(_stmt_lhs_name)
                                  and _stmt_lhs_name in self._variables
                                  and _stmt_lhs_name not in self._shared_scalars)
                    if is_ptr_var and op in (BinOp.ADD, BinOp.SUB):
                        # Pointer variable advance: scale integer offset by sizeof(*ptr).
                        scaled = self._scale_ptr_arith_offset(lhs.ty, rhs)
                        result = self._new_val("compound", lhs.ty)
                        self._emit(BinInst(result, op, lhs, scaled))
                        self._variables[_stmt_lhs_name] = result
                    else:
                        # Memory compound: load current value, compute, store back.
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
                        # Distinguish pointer-variable advance (p++) from
                        # array-element increment (arr[i]++).  For element
                        # increment the lhs is a computed address not tracked
                        # in _variables; for pointer advance it IS the tracked
                        # variable (or matches via _stmt_lhs_name).
                        update_name = _stmt_lhs_name or lhs.name
                        is_ptr_var = (update_name in self._variables
                                      and update_name not in self._shared_scalars)
                        if is_ptr_var:
                            # Pointer variable advance: p++ → p += sizeof(*p)
                            step = lhs.ty.pointee.size if isinstance(lhs.ty.pointee, ScalarTy) else 1
                            new_ptr = self._new_val(f"{lhs.name}_inc", lhs.ty)
                            self._emit(BinInst(new_ptr, op, lhs, Const(UINT64, step)))
                            self._variables[update_name] = new_ptr
                        else:
                            # Element address: arr[i]++ → load, add 1, store back
                            cur = self._new_val("cur", lhs.ty.pointee)
                            self._emit(LoadInst(cur, lhs))
                            result = self._new_val("compound", cur.ty)
                            self._emit(BinInst(result, op, cur, Const(cur.ty, 1)))
                            self._emit(StoreInst(addr=lhs, value=result))
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
            # Global device variable lvalue: g_counter = val; or g_arr[i] = val;
            # The variable lives in _global_consts as a SymbolRef, not _variables.
            # Return its address so the assignment handler can emit StoreInst.
            if (name not in self._variables and name in self._global_consts):
                cv = self._global_consts[name]
                if isinstance(cv, SymbolRef) and isinstance(cv.ty, PtrTy):
                    _nxt = self._toks[self._pos + 1] if self._pos + 1 < len(self._toks) else None
                    _assign_ops = (TokKind.ASSIGN, TokKind.PLUS_EQ, TokKind.MINUS_EQ,
                                   TokKind.STAR_EQ, TokKind.SLASH_EQ, TokKind.PERCENT_EQ,
                                   TokKind.AMP_EQ, TokKind.PIPE_EQ, TokKind.CARET_EQ,
                                   TokKind.LSHIFT_EQ, TokKind.RSHIFT_EQ,
                                   TokKind.LBRACKET, TokKind.DOT)
                    if _nxt and _nxt.kind in _assign_ops:
                        self._advance()  # consume ident
                        addr_val = self._new_val(f"{cv.sym_name}_ptr", cv.ty)
                        self._emit(GlobalAddrInst(addr_val, cv.sym_name, cv.ty.addr_space))
                        # g_arr[idx] = val — consume subscript and compute element address
                        if self._at(TokKind.LBRACKET):
                            self._advance()  # consume '['
                            idx_expr = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            elem_ty = cv.ty.pointee
                            # 2D array: if row stride registered and a second '[' follows,
                            # multiply by row_stride (not elem_size) and keep as pointer.
                            row_stride = self._array_row_strides.get(cv.sym_name)
                            if row_stride is not None and self._at(TokKind.LBRACKET):
                                idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, idx_expr,
                                                   Const(idx_ty, row_stride)))
                                row_addr = self._new_val("raddr", cv.ty)
                                self._emit(BinInst(row_addr, BinOp.ADD, addr_val, scaled))
                                # Consume second subscript [col]
                                self._advance()  # consume '['
                                col_expr = self._parse_expr()
                                self._expect(TokKind.RBRACKET)
                                elem_size = elem_ty.size if isinstance(elem_ty, ScalarTy) else 8
                                if elem_size != 1:
                                    col_ty = col_expr.ty if isinstance(col_expr, Value) else INT32
                                    scaled2 = self._new_val("scale", col_ty)
                                    self._emit(BinInst(scaled2, BinOp.MUL, col_expr,
                                                       Const(col_ty, elem_size)))
                                    col_expr = scaled2
                                elem_addr = self._new_val("addr", cv.ty)
                                self._emit(BinInst(elem_addr, BinOp.ADD, row_addr, col_expr))
                                cur_addr = elem_addr
                                cur_ty = elem_ty
                            else:
                                elem_size = elem_ty.size if isinstance(elem_ty, ScalarTy) else elem_ty.size
                                if elem_size != 1:
                                    idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                                    scaled = self._new_val("scale", idx_ty)
                                    self._emit(BinInst(scaled, BinOp.MUL, idx_expr,
                                                       Const(idx_ty, elem_size)))
                                    idx_expr = scaled
                                elem_addr = self._new_val("addr", cv.ty)
                                self._emit(BinInst(elem_addr, BinOp.ADD, addr_val, idx_expr))
                                cur_addr = elem_addr
                                cur_ty = elem_ty
                            while self._at(TokKind.DOT) and isinstance(cur_ty, StructTy):
                                self._advance()  # consume '.'
                                field = self._expect(TokKind.IDENT).value
                                field_off = cur_ty.field_offset(field)
                                field_ty = cur_ty.field_type(field)
                                field_ptr_ty = PtrTy(field_ty, cv.ty.addr_space)
                                field_addr = self._new_val("faddr", field_ptr_ty)
                                self._emit(BinInst(field_addr, BinOp.ADD, cur_addr,
                                                   Const(INT32, field_off)))
                                cur_addr = field_addr
                                cur_ty = field_ty
                            return cur_addr
                        # g_struct.field[.subfield...] = val — compute field address
                        if self._at(TokKind.DOT) and isinstance(cv.ty.pointee, StructTy):
                            cur_addr = addr_val
                            cur_ty = cv.ty.pointee
                            addr_space = cv.ty.addr_space
                            while self._at(TokKind.DOT) and isinstance(cur_ty, StructTy):
                                self._advance()  # consume '.'
                                field = self._expect(TokKind.IDENT).value
                                field_off = cur_ty.field_offset(field)
                                field_ty = cur_ty.field_type(field)
                                field_ptr_ty = PtrTy(field_ty, addr_space)
                                field_addr = self._new_val("faddr", field_ptr_ty)
                                self._emit(BinInst(field_addr, BinOp.ADD, cur_addr,
                                                   Const(INT32, field_off)))
                                cur_addr = field_addr
                                cur_ty = field_ty
                            return cur_addr
                        return addr_val
            if name in self._variables:
                var = self._variables[name]
                if isinstance(var.ty, PtrTy):
                    self._advance()
                    # ptr->field or ptr.field lvalue: compute address for StoreInst
                    # __shared__ vars are PtrTy; dot and arrow both mean field access.
                    if ((self._at(TokKind.ARROW) or self._at(TokKind.DOT))
                            and isinstance(var.ty.pointee, StructTy)):
                        self._advance()  # consume '->'
                        member = self._expect(TokKind.IDENT).value
                        sty = var.ty.pointee
                        arr_info = self._struct_field_arrays.get(sty.name, {})
                        if member in arr_info and self._at(TokKind.LBRACKET):
                            # ptr->arr[i] — array member field; compute element address
                            self._advance()  # consume '['
                            idx_expr = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            base_name = f"{member}_0"
                            base_off = sty.field_offset(base_name)
                            field_ty = sty.field_type(base_name)
                            base_addr = self._new_val("faddr", PtrTy(field_ty, var.ty.addr_space))
                            self._emit(BinInst(base_addr, BinOp.ADD, var, Const(INT32, base_off)))
                            elem_size = field_ty.size
                            if elem_size != 1:
                                idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, idx_expr, Const(idx_ty, elem_size)))
                                idx_expr = scaled
                            addr = self._new_val("faddr", PtrTy(field_ty, var.ty.addr_space))
                            self._emit(BinInst(addr, BinOp.ADD, base_addr, idx_expr))
                        else:
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
                        # ptr->ptr_field[idx] = v: the field is a pointer-typed member.
                        # addr currently points to the pointer field itself; load it to
                        # get the actual pointer, then index into it for the element address.
                        if (self._at(TokKind.LBRACKET) and isinstance(addr.ty, PtrTy)
                                and isinstance(addr.ty.pointee, PtrTy)):
                            ptr_val = self._new_val("pfield", addr.ty.pointee)
                            self._emit(LoadInst(ptr_val, addr))
                            self._advance()  # consume '['
                            idx_expr = self._parse_expr()
                            self._expect(TokKind.RBRACKET)
                            elem_ty = addr.ty.pointee.pointee
                            elem_size = elem_ty.size
                            if elem_size != 1:
                                idx_ty = idx_expr.ty if isinstance(idx_expr, Value) else INT32
                                scaled = self._new_val("scale", idx_ty)
                                self._emit(BinInst(scaled, BinOp.MUL, idx_expr, Const(idx_ty, elem_size)))
                                idx_expr = scaled
                            elem_addr = self._new_val("addr",
                                PtrTy(elem_ty, addr.ty.pointee.addr_space))
                            self._emit(BinInst(elem_addr, BinOp.ADD, ptr_val, idx_expr))
                            addr = elem_addr
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

    def _declare_local(self, name: str) -> None:
        """Register 'name' as declared in the current block scope (if inside one).

        Called at every variable-declaration site.  When the enclosing block
        ends, any name recorded here is either restored to its outer binding
        (if it shadowed an outer variable) or removed from _variables (if it
        was brand-new).  This prevents inner-scope declarations from corrupting
        outer-scope / loop-carried variables via the loop writeback mechanism.
        """
        if self._scope_locals_stack:
            self._scope_locals_stack[-1].add(name)

    def _parse_stmt_or_block(self):
        if self._match(TokKind.LBRACE):
            # Save outer bindings so we can restore shadowed names on exit.
            outer_bindings = dict(self._variables)
            self._scope_locals_stack.append(set())
            while not self._match(TokKind.RBRACE):
                self._parse_stmt()
            inner_decls = self._scope_locals_stack.pop()
            for name in inner_decls:
                if name in outer_bindings:
                    # Re-declaration (shadowing): restore the outer binding so
                    # the loop writeback doesn't mistake the inner Value for a
                    # modification of the outer variable.
                    self._variables[name] = outer_bindings[name]
                elif name in self._variables:
                    # Purely inner-scope variable: remove when scope ends.
                    del self._variables[name]
        else:
            self._parse_stmt()

    # -- Top-level parsing ---------------------------------------------------

    def _parse_kernel(self):
        self._expect(TokKind.KW_GLOBAL)
        # Skip optional __launch_bounds__(maxThreads, minBlocks) and
        # __attribute__((key(val,...))) — may appear before or after return type.
        # Capture __launch_bounds__ args for PTX .maxntid / .minnctapersm.
        _launch_bounds = [None, None]  # [maxThreadsPerBlock, minBlocksPerMP]
        def _skip_paren_qualifier(self):
            if self._at(TokKind.IDENT) and self._peek().value in (
                    '__launch_bounds__', '__attribute__', '__noinline__',
                    '__forceinline__', '__inline__'):
                qual_name = self._peek().value
                self._advance()
                # Consume any trailing ((...)) argument list
                if self._at(TokKind.LPAREN):
                    depth = 1
                    self._advance()
                    if qual_name == '__launch_bounds__':
                        # Parse first arg: maxThreadsPerBlock
                        if self._peek().kind in (TokKind.INT_LIT, TokKind.IDENT):
                            try:
                                _launch_bounds[0] = int(self._peek().value)
                            except (ValueError, TypeError):
                                pass
                            self._advance()
                            # Optional second arg: minBlocksPerMP
                            if self._match(TokKind.COMMA):
                                if self._peek().kind in (TokKind.INT_LIT, TokKind.IDENT):
                                    try:
                                        _launch_bounds[1] = int(self._peek().value)
                                    except (ValueError, TypeError):
                                        pass
                                    self._advance()
                        # Consume remaining tokens until matching ')'
                        depth = 1
                        while depth > 0:
                            if self._peek().kind == TokKind.LPAREN: depth += 1
                            if self._peek().kind == TokKind.RPAREN: depth -= 1
                            self._advance()
                    else:
                        while depth > 0:
                            if self._peek().kind == TokKind.LPAREN: depth += 1
                            if self._peek().kind == TokKind.RPAREN: depth -= 1
                            self._advance()
        # Allow a sequence of qualifiers (e.g. __attribute__ __launch_bounds__)
        while (self._at(TokKind.IDENT) and self._peek().value in (
                '__launch_bounds__', '__attribute__', '__noinline__',
                '__forceinline__', '__inline__')):
            _skip_paren_qualifier(self)
        ret_ty = self._parse_type()  # should be void
        # Also skip qualifiers that appear after the return type
        while (self._at(TokKind.IDENT) and self._peek().value in (
                '__launch_bounds__', '__attribute__', '__noinline__',
                '__forceinline__', '__inline__')):
            _skip_paren_qualifier(self)
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
                pname = self._expect_ident().value
                # Array parameter: int arr[N] decays to int *arr in C.
                if self._at(TokKind.LBRACKET):
                    self._advance()  # consume '['
                    while not self._at(TokKind.RBRACKET):
                        self._advance()
                    self._advance()  # consume ']'
                    if not isinstance(pty, PtrTy):
                        pty = PtrTy(pty, 0)
                # Skip trailing __attribute__((unused)) and similar GCC qualifiers
                while self._at(TokKind.IDENT) and self._peek().value == '__attribute__':
                    self._advance()
                    if self._at(TokKind.LPAREN):
                        depth = 1
                        self._advance()
                        while depth > 0 and not self._at(TokKind.EOF):
                            if self._peek().kind == TokKind.LPAREN: depth += 1
                            elif self._peek().kind == TokKind.RPAREN: depth -= 1
                            self._advance()
                params.append(KernelParam(pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

        self._kernel = Kernel(name=name, params=params)
        # Attach __launch_bounds__ info if present
        if _launch_bounds[0] is not None:
            self._kernel._launch_bounds = tuple(
                x for x in _launch_bounds if x is not None)
        self._variables = {}
        self._block_count = 0
        # Clear inline struct return fields — this dict is keyed by Value id, and
        # each kernel resets its id counter to 0, so ids from a prior kernel would
        # alias a fresh param sentinel or return dest created in this kernel.
        self._inline_struct_return_fields = {}
        self._copy_chain_global = {}

        # Load kernel parameters into variables.  Struct-by-value params
        # are flattened to one ParamInst per scalar/pointer leaf field;
        # the codegen mirrors this by emitting one .param per leaf with
        # the compound `paramname_fieldname` naming.  Matches the
        # Itanium-style ABI nvcc uses for small POD structs.
        #
        # Dedup by name: FORGE-emitted code carries both `s: span<T>` and
        # an explicit `s_len: u64` for proof obligations (with `requires
        # s_len == s.len`).  Once flattened the names collide; the first
        # occurrence wins.  Matches FORGE's own codegen_cuda.ml.
        entry = self._new_block("entry")
        self._cur_block = entry
        self._lazy_params = {}
        _seen_param_names: set[str] = set()
        for i, p in enumerate(params):
            if isinstance(p.ty, StructTy):
                # Register a struct sentinel so dot-access discovers it
                if p.name not in self._variables:
                    sent = self._new_val(p.name, p.ty)
                    self._variables[p.name] = sent

                def _flat_struct_params(prefix: str, sty: 'StructTy',
                                        slot_idx: int) -> int:
                    """Emit one ParamInst per leaf field, returning the
                    next ParamInst slot index.  Sub-structs recurse.
                    Skip names already bound to dedup struct/primitive
                    name collisions."""
                    for _pfname, _pfty in sty.fields:
                        _key = f"{prefix}_{_pfname}"
                        if _key in _seen_param_names:
                            continue
                        if isinstance(_pfty, StructTy):
                            _sent = self._new_val(_key, _pfty)
                            self._variables[_key] = _sent
                            _seen_param_names.add(_key)
                            slot_idx = _flat_struct_params(_key, _pfty,
                                                            slot_idx)
                        else:
                            _pfval = self._new_val(_key, _pfty)
                            self._emit(ParamInst(_pfval, slot_idx, _key))
                            self._variables[_key] = _pfval
                            _seen_param_names.add(_key)
                            slot_idx += 1
                    return slot_idx
                _flat_struct_params(p.name, p.ty, i)
            else:
                if p.name in _seen_param_names:
                    continue
                val = self._new_val(p.name, p.ty)
                self._emit(ParamInst(val, i, p.name))
                self._variables[p.name] = val
                _seen_param_names.add(p.name)

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
                pname = self._expect_ident().value
                # Array parameter: int arr[N] decays to int *arr in C.
                # Consume optional [N] and convert type to pointer.
                if self._at(TokKind.LBRACKET):
                    self._advance()  # consume '['
                    while not self._at(TokKind.RBRACKET):
                        self._advance()  # consume size/expr tokens
                    self._advance()  # consume ']'
                    if not isinstance(pty, PtrTy):
                        pty = PtrTy(pty, 0)
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

    def _compile_device_func(self, dfunc_info, mod):
        """Compile a recursive __device__ function into IR (DeviceFunction).

        Instead of inlining, this builds a real function with BasicBlocks,
        emitting PTX .func that can be called via call.uni.
        """
        from .parser import ParseError  # noqa: avoid circular at module level
        from ..ir.nodes import DeviceFunction, KernelParam, BasicBlock, ParamInst, RetTerm

        name = dfunc_info['name']
        ret_ty = dfunc_info['ret_ty']
        params_info = dfunc_info['params']  # list of (pname, pty) tuples

        # Save ALL parser state
        saved_pos = self._pos
        saved_vars = dict(self._variables)
        saved_kernel = self._kernel
        saved_block_count = self._block_count
        saved_cur_block = self._cur_block
        saved_inline_target = self._inline_return_target
        saved_scope_stack = self._scope_locals_stack

        # Create the DeviceFunction IR node
        func_params = [KernelParam(pname, pty) for pname, pty in params_info]
        func = DeviceFunction(name=name, ret_ty=ret_ty, params=func_params)

        # Set up parser state for this function (duck-typed as Kernel)
        self._kernel = func
        self._variables = {}
        self._block_count = 0
        self._inline_return_target = None
        self._scope_locals_stack = []

        # Create entry block
        entry = self._new_block("entry")
        self._cur_block = entry

        # Emit ParamInst for each parameter
        for i, (pname, pty) in enumerate(params_info):
            val = self._new_val(pname, pty)
            self._emit(ParamInst(val, i, pname))
            self._variables[pname] = val

        # Parse the function body
        self._pos = dfunc_info['body_start']
        self._expect(TokKind.LBRACE)
        while not self._at(TokKind.RBRACE) and not self._at(TokKind.EOF):
            self._parse_stmt()
        if self._at(TokKind.RBRACE):
            self._advance()

        # Ensure the last block has a terminator
        if self._cur_block.terminator is None:
            self._cur_block.terminator = RetTerm()

        # Restore parser state
        self._pos = saved_pos
        self._variables = saved_vars
        self._kernel = saved_kernel
        self._block_count = saved_block_count
        self._cur_block = saved_cur_block
        self._inline_return_target = saved_inline_target
        self._scope_locals_stack = saved_scope_stack

        return func

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
        self._recursive_device_funcs = set()

        # --- Pass 1: register device functions, structs, typedefs, enums, constants ---
        # Defer kernel parsing to Pass 2 so recursion detection can run first.
        _deferred_kernel_positions = []

        while not self._at(TokKind.EOF):
            # Skip leading storage class qualifiers (static, inline) before __global__/__device__
            while self._at(TokKind.KW_STATIC):
                self._advance()
            # File-scope `(static) const TYPE NAME = VALUE;` — FORGE-emitted
            # CUDA wrappers (cuda/forge/*_forge.cu) use this form for M31_P
            # and similar field-prime constants.  Register the value in
            # _global_consts so identifier lookups in expressions resolve it.
            # Only fold a constant if the initializer is a Const expression;
            # complex initializers fall through and the declaration is
            # skipped.  Save/restore self._pos so a parse failure here
            # doesn't lose tokens for the dispatch below.
            if self._at(TokKind.KW_CONST):
                _saved_pos = self._pos
                self._advance()  # consume 'const'
                try:
                    _ty = self._parse_type_with_ptr()
                    if self._at(TokKind.IDENT):
                        _name = self._advance().value
                        if self._match(TokKind.ASSIGN):
                            _val_op = self._parse_assign_expr()
                            if isinstance(_val_op, Const):
                                self._global_consts[_name] = _val_op
                            # consume rest of statement up to ';'
                            while (not self._at(TokKind.SEMI)
                                   and not self._at(TokKind.EOF)):
                                self._advance()
                            self._match(TokKind.SEMI)
                            continue
                except Exception:
                    pass
                # Couldn't parse cleanly — rewind and let the normal
                # dispatch handle whatever's there.
                self._pos = _saved_pos
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
                # Peek ahead to distinguish forward declaration (__global__ foo(...);)
                # from full definition (__global__ foo(...) { ... }).
                # Forward decls hit SEMI before LBRACE; skip them without recording.
                _lookahead = self._pos + 1
                _is_fwd_decl = False
                while _lookahead < len(self._toks):
                    _lk = self._toks[_lookahead].kind
                    if _lk == TokKind.LBRACE:
                        break
                    if _lk == TokKind.SEMI:
                        _is_fwd_decl = True
                        break
                    _lookahead += 1
                if _is_fwd_decl:
                    # Forward declaration — skip to SEMI and discard
                    while not self._at(TokKind.SEMI) and not self._at(TokKind.EOF):
                        self._advance()
                    if self._at(TokKind.SEMI):
                        self._advance()
                else:
                    # Full definition — defer kernel parsing, record position, skip body
                    _deferred_kernel_positions.append(self._pos)
                    while not self._at(TokKind.LBRACE) and not self._at(TokKind.EOF):
                        self._advance()
                    if self._at(TokKind.LBRACE):
                        depth = 1
                        self._advance()
                        while depth > 0 and not self._at(TokKind.EOF):
                            if self._peek().kind == TokKind.LBRACE: depth += 1
                            elif self._peek().kind == TokKind.RBRACE: depth -= 1
                            self._advance()
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
                    is_volatile = False
                    while self._at(TokKind.KW_DEVICE) or self._at(TokKind.KW_STATIC):
                        self._advance()
                    if self._at(TokKind.KW_CONSTANT):
                        is_const = True
                        self._advance()
                    # Consume volatile before type so _parse_type_with_ptr sees a clean type.
                    # For non-pointer vars (volatile int g_flag), volatile would otherwise
                    # be consumed by _parse_type_with_ptr but never propagated to PtrTy.
                    if self._at(TokKind.KW_VOLATILE):
                        is_volatile = True
                        self._advance()
                    ty = self._parse_type_with_ptr()
                    # If _parse_type_with_ptr produced a PtrTy (volatile T * case), inherit
                    # its volatile flag; otherwise use the is_volatile we captured above.
                    if isinstance(ty, PtrTy) and ty.volatile:
                        is_volatile = True
                    name = self._expect(TokKind.IDENT).value
                    # Optional array size [N] or multi-dim [N][M]...
                    count = 1
                    if self._match(TokKind.LBRACKET):
                        sz_op = self._parse_assign_expr()
                        d0 = int(sz_op.value) if isinstance(sz_op, Const) else 1
                        count = d0
                        self._expect(TokKind.RBRACKET)
                        # Additional dimensions: [M][K]...
                        inner_dims = []
                        while self._at(TokKind.LBRACKET):
                            self._advance()
                            dim_op = self._parse_assign_expr()
                            dim = int(dim_op.value) if isinstance(dim_op, Const) else 1
                            inner_dims.append(dim)
                            count *= dim
                            self._expect(TokKind.RBRACKET)
                        if inner_dims:
                            elem_size = ty.size if isinstance(ty, ScalarTy) else 8
                            inner_prod = 1
                            for d in inner_dims:
                                inner_prod *= d
                            self._array_row_strides[name] = inner_prod * elem_size
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
                    ptr_ty = PtrTy(ty, addr, volatile=is_volatile)
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

        # --- Recursion detection ---
        if self._device_funcs:
            all_dfunc_names = set(self._device_funcs.keys())
            call_graph = {}
            for fname, finfo in self._device_funcs.items():
                callees = set()
                for i in range(finfo['body_start'], finfo['body_end']):
                    if i < len(self._toks):
                        t = self._toks[i]
                        if t.kind == TokKind.IDENT and t.value in all_dfunc_names:
                            callees.add(t.value)
                call_graph[fname] = callees

            WHITE, GRAY, BLACK = 0, 1, 2
            color = {n: WHITE for n in call_graph}
            on_cycle = set()

            def _dfs_cycle(node, path):
                color[node] = GRAY
                path.append(node)
                for nxt in call_graph.get(node, ()):
                    if nxt not in color:
                        continue
                    if color[nxt] == GRAY:
                        idx = path.index(nxt)
                        on_cycle.update(path[idx:])
                    elif color[nxt] == WHITE:
                        _dfs_cycle(nxt, path)
                path.pop()
                color[node] = BLACK

            for node in call_graph:
                if color[node] == WHITE:
                    _dfs_cycle(node, [])
            self._recursive_device_funcs = on_cycle

            # Compile recursive device functions into IR
            for fname in self._recursive_device_funcs:
                dfunc = self._device_funcs[fname]
                compiled = self._compile_device_func(dfunc, mod)
                mod.device_functions.append(compiled)

        # --- Pass 2: parse deferred kernels ---
        for kpos in _deferred_kernel_positions:
            self._pos = kpos
            mod.kernels.append(self._parse_kernel())

        return mod


def parse(source: str) -> Module:
    """Parse CUDA-subset C source into an IR Module."""
    tokens = lex(source)
    return Parser(tokens).parse_module()
