# OpenCUDA Language Specification

**Version:** Corresponds to OpenCUDA as of 2026-03-22
**PTX Target:** PTX ISA 9.0, SM_120 (NVIDIA Blackwell / RTX 5090)

---

## 1. Overview

OpenCUDA is a pure-Python compiler that translates a subset of CUDA C to PTX assembly. It targets NVIDIA Blackwell (SM_120 / RTX 5090) and uses PTX ISA version 9.0 with 64-bit addressing throughout. No NVIDIA toolchain is required for PTX emission; `ptxas` or OpenPTXas is only needed to produce a cubin binary or to run the PTX validation test suite.

The supported language is a strict subset of CUDA C. It covers the constructs needed for data-parallel compute kernels: arithmetic, control flow, memory access, shared memory, atomics, warp shuffles, device function inlining, and the preprocessor `#define` directive. C++ features (templates, classes, namespaces, exceptions) are not supported.

---

## 2. Types

### 2.1 Scalar types

| C syntax | Internal type | PTX type | Notes |
|---|---|---|---|
| `void` | `VOID` | `u32` (unused) | Valid only as kernel return type |
| `bool` | `BOOL` | `u32` | Parsed; stored as 32-bit integer |
| `char` | `INT8` | `u32` | No dedicated PTX 8-bit arithmetic; treated as 32-bit |
| `unsigned char` | `UINT8` | `u32` | Same as char |
| `short` | `INT16` | `u32` | No dedicated PTX 16-bit arithmetic; treated as 32-bit |
| `unsigned short` | `UINT16` | `u32` | Same as short |
| `int` | `INT32` | `s32` | Default integer type |
| `unsigned` / `unsigned int` | `UINT32` | `u32` | |
| `long` | `INT32` | `s32` | Treated as 32-bit (not 64-bit) |
| `long long` | `INT64` | `s64` | |
| `unsigned long long` | `UINT64` | `u64` | |
| `float` | `FLOAT` | `f32` | |
| `double` | `DOUBLE` | `f64` | |
| `half` | `HALF` | `f16` | Load/store uses `.b16`; arithmetic emits native `add.f16`, `sub.f16`, `mul.f16`, `div.approx.f16` — see Section 11 |

Notes on `char` and `short`: these types are parsed and represented in the IR type system, but there is no dedicated 8-bit or 16-bit PTX arithmetic. All scalar operations on these types fall back to `u32` in PTX. They are included for source compatibility but should not be relied on for bit-exact narrow arithmetic.

### 2.2 Pointer types

Pointers are 64-bit (8 bytes) on SM_120. All pointer arithmetic is performed using `u64`.

```c
float *p;         // pointer to float (global address space by default)
int *q;           // pointer to int
struct Foo *s;    // pointer to struct
```

Pointer declarations accept `const` and `__restrict__` qualifiers, which are parsed and silently discarded (no aliasing analysis is performed).

### 2.3 Address spaces

| Address space | Keyword | PTX qualifier | Notes |
|---|---|---|---|
| Generic | (default) | (none) | Default for kernel parameters and local scalars |
| Global | `__global__` | `.global` | Global memory; used for kernel parameter pointers |
| Shared | `__shared__` | `.shared` | On-chip shared memory; declared with `__shared__ T name[N]` |
| Local | (internal) | `.local` | Used internally by `printf` valist temporaries |
| Const (read-only) | (via `__ldg`) | `.global.nc` | Non-coherent load cache; activated by `__ldg(ptr)` |

### 2.4 Struct types

```c
struct Vec2 { float x; float y; };
typedef struct Vec2 Vec2;
```

Structs are defined at file scope using `struct Name { field_decls };`. Field offsets are computed with natural alignment (each field aligned to its own size). `typedef struct Name Name;` is supported.

### 2.5 Type parsing rules

- `const` is recognized as a qualifier before or after the type and after each pointer star, and is silently stripped. There is no const-correctness enforcement.
- `__restrict__` immediately following the pointer declarator is recognized and silently stripped.
- Pointer stars are right-associative in the type parser: `int * const *` parses as pointer to pointer to int (the middle `const` is stripped).

---

## 3. Operators

### 3.1 Arithmetic operators

| Operator | Description | PTX instruction | Status |
|---|---|---|---|
| `+` | Addition | `add.{s32,u32,s64,u64,f32,f64}` or `add.u64` for pointers | Supported |
| `-` | Subtraction | `sub.*` | Supported |
| `*` | Multiplication | `mul.lo.*` (integers), `mul.*` (floats) | Supported |
| `/` | Division | `div.*` | Supported (integers: `div.s32`/`div.u32`; floats: `div.f32`/`div.f64`) |
| `%` | Modulo/remainder | `rem.*` | Supported |
| `-` (unary) | Negation | `sub.* 0, x` | Supported |

### 3.2 Bitwise operators

| Operator | Description | PTX instruction | Status |
|---|---|---|---|
| `&` | Bitwise AND | `and.b32` / `and.b64` | Supported |
| `\|` | Bitwise OR | `or.b32` / `or.b64` | Supported |
| `^` | Bitwise XOR | `xor.b32` / `xor.b64` | Supported |
| `~` | Bitwise NOT | `xor.b32 dest, src, -1` | Supported |
| `<<` | Left shift | `shl.b32` / `shl.b64` | Supported |
| `>>` | Right shift | `shr.b32` / `shr.b64` | Supported |

Note: bitwise operators use `.b32`/`.b64` PTX type qualifiers regardless of whether the operands are declared `signed` or `unsigned`.

### 3.3 Compound assignment operators

All compound assignment operators are supported: `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`.

For pointer lvalues (array elements), compound assignment correctly performs a load-modify-store. Note: at statement level, only `+=`, `-=`, and `*=` emit the full load-modify-store sequence for array elements; other compound assignments (`/=`, `%=`, `&=`, etc.) behave correctly only on scalar (register) variables within a statement context. All ten operators work correctly when used in expression context.

### 3.4 Increment and decrement

| Operator | Form | Status |
|---|---|---|
| `++` | Post-increment (prefix not supported) | Supported (postfix only) |
| `--` | Post-decrement (prefix not supported) | Supported (postfix only) |

Pre-increment (`++i`) is not supported. Use `i += 1` or `i = i + 1` instead.

### 3.5 Comparison operators

| Operator | PTX instruction | Status |
|---|---|---|
| `==` | `setp.eq.*` | Supported |
| `!=` | `setp.ne.*` | Supported |
| `<` | `setp.lt.*` | Supported |
| `<=` | `setp.le.*` | Supported |
| `>` | `setp.gt.*` | Supported |
| `>=` | `setp.ge.*` | Supported |

All six comparison operators produce a predicate register (`.pred`).

### 3.6 Logical operators

| Operator | Implementation | Status |
|---|---|---|
| `&&` | `and.b32` on integer values (short-circuit not guaranteed) | Supported |
| `\|\|` | `or.b32` on integer values (short-circuit not guaranteed) | Supported |
| `!` | `setp.eq.s32 pred, src, 0` | Supported |

Logical `&&` and `||` are lowered to bitwise AND/OR on integer values. Short-circuit evaluation is not guaranteed; both operands are always evaluated.

### 3.7 Ternary operator

```c
result = cond ? true_expr : false_expr;
```

Supported. Lowered to a conditional branch into two blocks followed by a merge block. The result type is inferred from the true expression.

### 3.8 Cast operator

```c
(float)i     // int → float
(int)f       // float → int (truncates toward zero)
(double)f    // float → double
(float)h     // half → float
```

Explicit casts are supported and emit `cvt` instructions with appropriate rounding modes:
- Integer to float: `cvt.rn.f32.s32` (round to nearest)
- Float to integer: `cvt.rzi.s32.f32` (round toward zero / truncate)
- Float narrowing (e.g., double to float): `cvt.rn.f32.f64`
- Same-width float conversion: no rounding modifier

Pointer casts (`(float *)p`) are supported syntactically.

### 3.8 Address-of operator

`&expr` is supported in a limited form: `&ptr[index]` computes the address of an array element without loading it. This is the primary supported use case (e.g., `__ldg(&in[tid])`).

General address-of for scalar variables (taking the address of a register variable) is not supported.

### 3.9 Dereference operator

`*ptr` loads the value at the pointer. Supported when `ptr` is a `PtrTy` value.

### 3.10 Type promotion rules

When two operands of different types are combined in a binary expression, the result type is determined by:
1. If either operand is a floating-point type, the result is `float` (no double promotion from float operands — double is only preserved if a `double` variable is directly involved).
2. If either operand is a pointer, the result is the pointer type.
3. If either operand is 64-bit integer, the result is that 64-bit type.
4. Otherwise, the result is the type of the left operand.

---

## 4. Statements

| Statement | Example | Status |
|---|---|---|
| Variable declaration | `int x = 0;` | Supported |
| `const` declaration | `const int N = 256;` | Supported (`const` stripped; treated as a regular variable) |
| Assignment | `x = expr;` | Supported |
| Array element assignment | `ptr[i] = expr;` | Supported |
| Compound assignment | `x += y;` | Supported (all ten operators) |
| Array compound assignment | `ptr[i] += y;` | Partially supported (see Section 3.3) |
| `if` / `if-else` | `if (cond) { ... } else { ... }` | Supported |
| `for` loop | `for (int i = 0; i < n; i++) { ... }` | Supported |
| `while` loop | `while (cond) { ... }` | Supported |
| `do-while` loop | `do { ... } while (cond);` | Supported |
| `break` | `break;` | Supported (exits innermost loop or switch) |
| `continue` | `continue;` | Supported (skips to loop increment/condition) |
| `switch` / `case` / `default` | `switch (x) { case 1: ...; break; default: ...; }` | Supported (lowered to comparison chain) |
| `return` (void) | `return;` | Supported |
| `return` (value) | `return expr;` | Supported in `__device__` functions; kernel `return` emits `ret;` |
| Array indexing | `ptr[i]` | Supported |
| Struct member access (pointer) | `ptr->field` | Supported |
| Struct member access (value) | `s.field` | Partially supported — see Section 8 |
| Expression statement | `f();` | Supported |

### 4.1 Variable scope

OpenCUDA does not implement lexical block scoping. All variables declared anywhere within a kernel (or inlined device function) share a single flat namespace. Redeclaring a variable in an inner block will shadow the outer declaration within the parser's variable map.

---

## 5. CUDA Built-ins

### 5.1 Thread and block indexing

All nine built-in index/dimension accessors are supported:

| C expression | PTX instruction | Result type |
|---|---|---|
| `threadIdx.x` | `mov.u32 dest, %tid.x;` | `int` |
| `threadIdx.y` | `mov.u32 dest, %tid.y;` | `int` |
| `threadIdx.z` | `mov.u32 dest, %tid.z;` | `int` |
| `blockIdx.x` | `mov.u32 dest, %ctaid.x;` | `int` |
| `blockIdx.y` | `mov.u32 dest, %ctaid.y;` | `int` |
| `blockIdx.z` | `mov.u32 dest, %ctaid.z;` | `int` |
| `blockDim.x` | `mov.u32 dest, %ntid.x;` | `int` |
| `blockDim.y` | `mov.u32 dest, %ntid.y;` | `int` |
| `blockDim.z` | `mov.u32 dest, %ntid.z;` | `int` |

`gridDim` is not supported.

### 5.2 Synchronization

| Function | PTX instruction | Status |
|---|---|---|
| `__syncthreads()` | `bar.sync 0;` | Supported |

No other synchronization primitives (`__syncwarp`, `__threadfence`, `__threadfence_block`, `__threadfence_system`) are supported.

### 5.3 Memory qualifiers

| Qualifier | Where | Effect |
|---|---|---|
| `__global__` | Function | Marks the function as a GPU kernel entry point |
| `__shared__` | Local array declaration | Allocates array in `.shared` address space |
| `__device__` | Function | Marks function for inlining at call site |
| `__forceinline__` | Function | Treated identically to `__device__` |
| `__restrict__` | Parameter pointer | Parsed and silently ignored |
| `const` | Anywhere | Parsed and silently ignored |

`__shared__` arrays must have a compile-time integer literal size. Dynamically-sized shared memory (`extern __shared__`) is not supported.

### 5.4 Atomic operations

All nine CUDA atomic operations are supported:

| Function | PTX instruction | Description |
|---|---|---|
| `atomicAdd(addr, val)` | `atom.global.add.*` | Atomic add |
| `atomicSub(addr, val)` | `atom.global.add.*` (negated) | Atomic subtract (emitted as add) |
| `atomicMin(addr, val)` | `atom.global.min.*` | Atomic minimum |
| `atomicMax(addr, val)` | `atom.global.max.*` | Atomic maximum |
| `atomicAnd(addr, val)` | `atom.global.and.*` | Atomic bitwise AND |
| `atomicOr(addr, val)` | `atom.global.or.*` | Atomic bitwise OR |
| `atomicXor(addr, val)` | `atom.global.xor.*` | Atomic bitwise XOR |
| `atomicExch(addr, val)` | `atom.global.exch.*` | Atomic exchange |
| `atomicCAS(addr, cmp, val)` | `atom.global.cas.*` | Atomic compare-and-swap |

All atomics operate on global memory. The PTX type is inferred from the second argument: `f32` for float constants/values, `u32` otherwise. Return value (old value) is available.

### 5.5 Warp operations

| Function | PTX instruction | Description |
|---|---|---|
| `__shfl_sync(mask, val, lane)` | `shfl.sync.idx.b32` | Shuffle from lane (indexed) |
| `__shfl_up_sync(mask, val, delta)` | `shfl.sync.up.b32` | Shuffle from lower lane |
| `__shfl_down_sync(mask, val, delta)` | `shfl.sync.down.b32` | Shuffle from upper lane |
| `__shfl_xor_sync(mask, val, lane_mask)` | `shfl.sync.bfly.b32` | Butterfly shuffle |
| `__ballot_sync(mask, pred)` | `vote.sync.ballot.b32` | Warp vote / ballot |

All warp operations emit 32-bit (`b32`) PTX instructions. The membership mask argument is passed through; the clamp value is hardcoded to `31` in the `shfl.sync` encoding.

### 5.6 Read-only (non-coherent) cache load

```c
float val = __ldg(&ptr[i]);
```

`__ldg(ptr)` is supported. It emits `ld.global.nc.{type}` (non-coherent global load). The argument must be in the `&ptr[idx]` form — `__ldg` of an arbitrary pointer expression is not supported.

### 5.7 printf

```c
printf("format string", arg1, arg2, ...);
```

`printf` is supported and lowered to the PTX `vprintf` ABI:
- The format string must be a string literal (not a variable or expression).
- Arguments are widened for the variadic convention: `int`/`unsigned` → `u64`, `float` → `f64` (promoted to double).
- Supported argument types: integer types and `float`/`double`.
- String pointer arguments and `half` arguments are not supported.
- The return value of `printf` is discarded.
- Format string escape sequences `\n`, `\t`, `\r`, `\\`, `\"` are processed at compile time.

---

## 6. Preprocessor

OpenCUDA implements a minimal preprocessor with a single supported directive:

```c
#define NAME VALUE
```

Text substitution: replaces all occurrences of the identifier `NAME` in the source with the literal text `VALUE`. Substitution is performed on the raw source text before lexing (a simple regex replacement).

All other preprocessor directives are silently ignored:

| Directive | Status |
|---|---|
| `#define NAME VALUE` | Supported (text substitution) |
| `#define NAME` (no value) | Silently ignored |
| `#include` | Silently ignored |
| `#ifdef` / `#ifndef` / `#endif` | Silently ignored |
| `#if` / `#elif` / `#else` | Silently ignored |
| `#pragma` | Silently ignored |
| `#undef` | Silently ignored |

Function-like macros (`#define F(x) (x*x)`) are not supported; only object-like macros with a single replacement value work reliably.

---

## 7. Functions

### 7.1 `__global__` kernels

```c
__global__ void kernel_name(param_type param, ...) {
    // body
}
```

- Return type must be `void`.
- Parameters may be any supported type (scalar or pointer).
- Multiple `__global__` kernels may be defined in the same `.cu` file (compiled to a single PTX file).
- `__launch_bounds__(maxThreads, minBlocks)` is parsed and silently ignored.

### 7.2 `__device__` functions

```c
__device__ return_type func_name(param_type param, ...) {
    // body
}
```

- `__device__` functions are inlined at every call site. No separate PTX function is generated.
- Any return type is supported.
- Multiple return points within the function body are supported (via a merge block mechanism).
- Recursive calls are not supported.
- Variadic parameters are not supported.
- `__forceinline__` is treated identically to `__device__`.

Device function inlining mechanism: at each call site, the compiler replays the function body token stream, binding argument values to parameter names, and routes all `return expr;` statements to a shared merge block. The return value is communicated via a dedicated SSA value.

### 7.3 Unsupported function features

- Recursion (including mutual recursion)
- Variadic functions (`...`)
- Function pointers
- `extern "C"` or any C++ linkage specifiers
- `inline` (C99 keyword — use `__device__` or `__forceinline__`)

---

## 8. Struct Support

### 8.1 Definition and typedef

```c
struct Vec3 { float x; float y; float z; };
typedef struct Vec3 Vec3;
```

Struct definitions are supported at file scope (before any kernel). `typedef struct Name Name;` is supported. Forward declarations are not supported — the struct must be defined before use.

### 8.2 Field access

| Access pattern | Status | Notes |
|---|---|---|
| `ptr->field` (pointer to struct) | Supported | Computes `base + field_offset` and loads |
| `s.field` on a struct-typed Value | Partial | Struct value in a register has no backing memory; field offset is computed but the load may not be well-defined |
| Nested struct access | Not supported | |
| Struct arrays (`arr[i].field`) | Supported | `arr[i]` yields a pointer offset; then `->field` works |

The reliable access pattern is via pointer: pass `struct Foo *` as a kernel parameter and use `ptr[i].field` (which is syntactic sugar for `(ptr + i)->field`).

### 8.3 Field layout

Fields are laid out with natural alignment: each field is aligned to its own size (1, 2, 4, or 8 bytes). No explicit padding control is supported (`#pragma pack`, `__attribute__((packed))`).

---

## 9. Optimization Passes

The optimizer runs three passes in sequence after parsing, before code generation.

| Pass | Scope | Description |
|---|---|---|
| Loop unrolling | Per-kernel | Detects `for` loops with a compile-time constant trip count. Unrolls loops with trip count ≤ 16. Loop-carried variable chains are wired explicitly (output of iteration N is input of iteration N+1). |
| Constant folding | Per basic block | Evaluates `Const OP Const` at compile time. Includes strength reduction: integer multiply by a power of two is converted to a shift (`x * 8 → x << 3`). Also folds `x * 0 → 0`. |
| CSE (Common Subexpression Elimination) | Per basic block | Eliminates duplicate computations within a basic block. CSE key includes the destination type to prevent merging integer and float operations that share the same operand IDs. Also CSE's redundant `cvt` instructions. |
| CVT CSE (widen cache) | Per kernel | During code generation, `cvt.u64.u32` instructions for the same source register are deduplicated (emitted once and reused). |
| Linear scan register allocation | Per kernel | Compact register IDs using live interval analysis. Registers are grouped by type prefix (`r`, `rd`, `f`, `fd`, `h`, `p`) and allocated separately. No spilling — infinite register assumption. |

**Safety constraint:** Constant folding and CSE never propagate values across basic block boundaries. This prevents the loop writeback bug where an initializer value (e.g., `float sum = 0`) would be replaced by a constant in the loop body, making loop-carried updates invisible to the loop condition.

---

## 10. PTX Target

| Property | Value |
|---|---|
| PTX ISA version | 9.0 |
| Target architecture | `sm_120` (NVIDIA Blackwell / RTX 5090) |
| Address size | 64-bit |
| Pointer size | 8 bytes |

### 10.1 Register types

| PTX register type | Prefix | Used for |
|---|---|---|
| `.b32` | `r` | `int`, `unsigned int`, `char`, `short`, `bool`, and all narrow integer types |
| `.b64` | `rd` | `long long`, `unsigned long long`, all pointer types |
| `.f32` | `f` | `float` |
| `.f64` | `fd` | `double` |
| `.f16` | `h` | `half` |
| `.pred` | `p` | Comparison results, predicate registers |

### 10.2 Load/store type encoding

| C type | Load/store PTX type | Notes |
|---|---|---|
| `int` | `.s32` | |
| `unsigned int` | `.u32` | |
| `long long` | `.s64` | |
| `unsigned long long` | `.u64` | |
| `float` | `.f32` | |
| `double` | `.f64` | |
| `half` | `.b16` | PTX does not support `ld.f16`; half loads use `.b16` |
| pointer (any) | `.u64` | Pointer parameters loaded as `ld.param.u64` |

### 10.3 Kernel structure

Every kernel emits a self-contained PTX module:

```ptx
.version 9.0
.target sm_120
.address_size 64

.visible .entry kernel_name(
    .param .s32 param0,
    .param .u64 param1)
{
    .shared .f32 smem[256];     // if __shared__ used
    .reg .b32 %r<N>;
    .reg .b64 %rd<N>;
    .reg .f32 %f<N>;
    .reg .pred %p<N>;

    // body
    ret;
}
```

---

## 11. Known Limitations

### 11.1 half (`float16`) arithmetic

`half` variables load and store using `.b16` (since PTX lacks `ld.f16`). For arithmetic, the compiler **does** emit native `f16` PTX instructions:

- `add.f16`, `sub.f16`, `mul.f16` — emitted for `half + half`, `half - half`, `half * half`
- `div.approx.f16` — emitted for `half / half`

However, the practical limitation is that the type inference in `_result_type()` promotes mixed `half`/`float` expressions to `float`. To reliably perform half arithmetic, both operands must be declared `half`. The common pattern is to load `half`, cast to `float`, perform arithmetic in float, then cast back — which is what all the `half_*` test kernels do.

There is no `fma.f16` emission; fused multiply-add requires explicit `fmaf()` calls, which are not mapped by the compiler.

### 11.2 Device function limitations

- No recursion (including mutual recursion).
- No variadic parameters.
- Device function bodies are replayed by re-parsing the token stream at each call site; this means the body cannot contain directives that alter global parser state (e.g., struct definitions inside a device function body would be re-registered on each inline).

### 11.3 Integer division and remainder

Integer `div` and `rem` emit `div.s32`/`div.u32`/`rem.s32`/`rem.u32` PTX instructions. PTX does not natively support division on SM_120 — these instructions require ptxas or OpenPTXas to expand them to SASS sequences.

### 11.4 Register allocation

Register allocation uses a linear scan algorithm. There is no register spilling. If a kernel has more live values than available physical registers, the generated PTX may be invalid (ptxas will reject it). This is unlikely for typical kernels but possible for pathologically register-heavy code.

### 11.5 No separate compilation

All kernels and device functions must be in a single `.cu` file. There is no support for linking multiple translation units.

### 11.6 No dynamic memory

`malloc`, `free`, `new`, `delete`, `cudaMalloc` calls in device code are not parsed. These functions are not available on the GPU without the device runtime library, which is not supported.

### 11.7 No C++ features

No templates, no classes (`class`), no namespaces, no exceptions, no references (`&` as a reference qualifier), no `auto` type inference, no range-based `for`, no lambda expressions.

### 11.8 No multi-dimensional arrays

Only 1D pointer-based arrays (`T *ptr`) are supported. 2D arrays (`T arr[M][N]`) and multi-dimensional pointer arrays (`T **`) are not supported.

### 11.9 printf limitations

- The format string must be a compile-time string literal. Passing a variable as the format argument silently produces an empty format string.
- Arguments must be integer or float types. String pointer arguments (`%s`), pointer arguments (`%p`), and `half` arguments are not supported.
- `printf` in device code uses the PTX `vprintf` ABI. Each call emits a local memory buffer for the argument list. Heavy printf usage will increase register and local memory pressure.

### 11.10 Struct limitations

- Direct struct value types (stack-allocated structs) have limited support for field access — the compiler computes field offsets but the resulting load may not be well-defined without backing memory. Use pointer-to-struct as the reliable pattern.
- Nested structs (struct containing another struct) are not supported.
- Struct arrays with field access work through the pointer path: `arr[i].field` is valid.
- Struct assignment (`s1 = s2;`) is not supported.
- Passing structs by value to device functions is not supported.

### 11.11 No texture or surface memory

`tex1D`, `tex2D`, `surf1Dread`, `surf1Dwrite`, and all other texture/surface memory operations are not supported.

### 11.12 No cooperative groups or tensor operations

`cooperative_groups::`, `wmma::`, and `nvcuda::` namespaces and their operations are not supported.

### 11.13 No warp-level primitives beyond shuffle/ballot

`__activemask()`, `__match_any_sync()`, `__match_all_sync()`, `__reduce_add_sync()`, and other warp-level intrinsics beyond the five shuffle/ballot functions listed in Section 5.5 are not supported.

### 11.14 Pre-increment/decrement not supported

`++i` and `--i` (prefix forms) are not recognized. Use `i += 1`, `i = i + 1`, or the postfix form `i++` (but note that postfix in expression context returns the old value).

### 11.15 `long` treated as 32-bit

`long` is parsed and treated as `int` (32-bit). Use `long long` for 64-bit integers.

---

## 12. Compilation Pipeline

```
CUDA C (.cu)
  → Preprocessor    #define NAME VALUE text substitution
  → Lexer           Regex tokenizer (lexer.py)
  → Parser          Recursive descent → SSA IR (parser.py)
  → Optimizer       Loop unrolling, constant folding, CSE (optimize.py / unroll.py)
  → Codegen         PTX 9.0 text emission (emit.py)
  → [OpenPTXas]     Optional: PTX → cubin binary
```

### 12.1 SSA IR

The compiler uses a Static Single Assignment intermediate representation. Every `Value` is assigned exactly once. Control flow is represented as a graph of `BasicBlock` objects, each ending with a terminator (`ret`, unconditional branch, or conditional branch). `PhiInst` nodes are present in the IR data model but are not currently emitted by the parser — the parser instead uses a mutable variable map with explicit write-back at loop backedges.

### 12.2 Module structure

A single `.cu` file compiles to a single PTX file containing one PTX module. Each `__global__` kernel becomes one `.visible .entry` in the PTX. All kernels in a file are emitted in the order they appear in the source.

---

## 13. CLI Usage

```bash
# Emit PTX to stdout
python -m opencuda kernel.cu --emit-ptx

# Compile to cubin (requires OpenPTXas)
python -m opencuda kernel.cu --out kernel.cubin

# Run all tests
pytest opencuda/tests/test_compiler.py -v

# Run a single test by kernel name
pytest opencuda/tests/test_compiler.py -v -k vector_add
```

The test suite contains two parametrized test functions, each run against all 33 `.cu` files in `tests/`:

- `test_parse_and_emit` — verifies PTX output contains `.version`, `.entry`, and `ret;`
- `test_ptxas_validates` — feeds PTX to NVIDIA's `ptxas` for SM_120 validation (requires NVIDIA CUDA toolkit)

---

## 14. Example

```c
// vector_add.cu — add two float arrays
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}
```

```bash
python -m opencuda tests/vector_add.cu --emit-ptx
```

Output (abbreviated):

```ptx
.version 9.0
.target sm_120
.address_size 64

.visible .entry vector_add(
    .param .u64 out,
    .param .u64 a,
    .param .u64 b,
    .param .s32 n)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<3>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [out];
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.s32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    ...
    ret;
}
```
