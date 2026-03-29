"""
OpenCUDA IR — SSA-based intermediate representation.

Three-address code in SSA form. Every value is assigned exactly once.
Control flow is represented as a graph of basic blocks.

This IR is target-independent — it doesn't know about SM_120, registers,
or SASS. The codegen layer lowers it to OpenPTXas instructions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum

from .types import Type, ScalarTy, PtrTy, INT32, UINT32, FLOAT, VOID


# ---------------------------------------------------------------------------
# Values (SSA virtual registers)
# ---------------------------------------------------------------------------

@dataclass
class Value:
    """An SSA value (virtual register)."""
    name: str
    ty: Type
    id: int = 0  # unique ID assigned during construction

    def __str__(self) -> str:
        return f"%{self.name}"


@dataclass
class Const:
    """A compile-time constant."""
    ty: Type
    value: Union[int, float]

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class SymbolRef:
    """Reference to a named module-level PTX symbol (__device__ / __constant__).

    Carries the PTX symbol name and its pointer type. Used as an address operand
    in LoadInst/StoreInst/AtomInst for direct symbol access (e.g. [sym_name]).
    """
    sym_name: str
    ty: object  # PtrTy

    @property
    def name(self) -> str:
        return self.sym_name


Operand = Union[Value, Const, 'SymbolRef']


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

class BinOp(Enum):
    ADD  = "add"
    SUB  = "sub"
    MUL  = "mul"
    DIV  = "div"
    MOD  = "mod"
    AND  = "and"
    OR   = "or"
    XOR  = "xor"
    SHL  = "shl"
    SHR  = "shr"


class CmpOp(Enum):
    EQ  = "=="
    NE  = "!="
    LT  = "<"
    LE  = "<="
    GT  = ">"
    GE  = ">="


@dataclass
class BinInst:
    """Binary arithmetic/logic: dest = lhs OP rhs."""
    dest: Value
    op: BinOp
    lhs: Operand
    rhs: Operand


@dataclass
class CmpInst:
    """Comparison: dest = lhs CMP rhs (result is bool/predicate)."""
    dest: Value
    op: CmpOp
    lhs: Operand
    rhs: Operand


@dataclass
class LoadInst:
    """Load from memory: dest = *addr."""
    dest: Value
    addr: Operand


@dataclass
class StoreInst:
    """Store to memory: *addr = value."""
    addr: Operand
    value: Operand


@dataclass
class CvtInst:
    """Type conversion: dest = (dest.ty)src."""
    dest: Value
    src: Operand


@dataclass
class CallInst:
    """Built-in function call: dest = func(args...)."""
    dest: Optional[Value]
    func: str  # "threadIdx.x", "blockIdx.x", "blockDim.x", "__syncthreads", etc.
    args: list[Operand] = field(default_factory=list)


@dataclass
class PhiInst:
    """SSA phi node: dest = phi(val_from_bb1, val_from_bb2, ...)."""
    dest: Value
    incoming: list[tuple[Operand, str]]  # (value, block_label)


@dataclass
class ParamInst:
    """Kernel parameter load: dest = kernel_param[index]."""
    dest: Value
    param_index: int
    param_name: str


@dataclass
class PrintfInst:
    """printf("fmt", args...) intrinsic — lowered to vprintf in PTX."""
    fmt: str          # processed format string (escapes resolved)
    args: list[Operand] = field(default_factory=list)


@dataclass(frozen=True)
class GlobalAddrInst:
    """Materialize the address of a named module-level symbol into a u64 register.

    PTX emission:
      .global symbol → cvta.to.global.u64 dest, sym_name;
      .const  symbol → cvta.to.const.u64  dest, sym_name;
    """
    dest: Value
    sym_name: str
    addr_space: object  # AddrSpace


Instruction = Union[BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                    CallInst, PhiInst, ParamInst, PrintfInst, GlobalAddrInst]


# ---------------------------------------------------------------------------
# Terminators (end of basic block)
# ---------------------------------------------------------------------------

@dataclass
class RetTerm:
    """Return from kernel."""
    pass


@dataclass
class BrTerm:
    """Unconditional branch."""
    target: str  # block label


@dataclass
class CondBrTerm:
    """Conditional branch: if cond goto true_bb else goto false_bb."""
    cond: Operand
    true_bb: str
    false_bb: str


Terminator = Union[RetTerm, BrTerm, CondBrTerm]


# ---------------------------------------------------------------------------
# Basic blocks and functions
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    label: str
    instructions: list[Instruction] = field(default_factory=list)
    terminator: Optional[Terminator] = None


@dataclass
class KernelParam:
    name: str
    ty: Type


@dataclass
class Kernel:
    """A GPU kernel function."""
    name: str
    params: list[KernelParam]
    blocks: list[BasicBlock] = field(default_factory=list)
    _next_id: int = 0

    def new_value(self, name: str, ty: Type) -> Value:
        v = Value(name, ty, self._next_id)
        self._next_id += 1
        return v

    @property
    def entry_block(self) -> BasicBlock:
        return self.blocks[0] if self.blocks else None


@dataclass
class Module:
    """A compilation unit (one .cu file)."""
    kernels: list[Kernel] = field(default_factory=list)
    # Module-level global/constant variable declarations:
    # each entry is (name: str, elem_ty: Type, count: int, addr_space: AddrSpace)
    global_vars: list = field(default_factory=list)
