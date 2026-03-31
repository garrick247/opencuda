# OpenCUDA

**Full CUDA C pipeline, fully open-source, validated on RTX 5090.**

OpenCUDA compiles CUDA C to PTX. [OpenPTXas](https://github.com/garrick99/openptxas) assembles PTX to SM_120 cubins. The resulting kernels run correctly on Blackwell hardware — without nvcc or ptxas.

**Just Python. Real cubins. Real GPU. Correct output.**

## The Proof

CUDA C compiled to working SM_120 cubins using only open-source Python tools.

**GPU-verified on RTX 5090:**

| Kernel | What it does | Status |
|--------|-------------|--------|
| `vector_add` | `out[i] = a[i] + b[i]` (float, multi-block) | **PASS** |
| `kernel_a` | `out[i] = in[i] * 2.0f` (float multiply by constant) | **PASS** |
| `increment` | `out[i] = in[i] + 1` (integer add) | **PASS** |
| `divergent_warp` | Predicated early exit, intra-warp divergence | **PASS** |
| `sel` | `out[i] = v > 0.5f ? 1.0f : 0.0f` (float ternary + bounds check) | **PASS** |

**No nvcc. No ptxas. No NVIDIA compiler.**

## How It Works

```
CUDA C source (.cu)
    |  OpenCUDA (Python)
    v
PTX assembly
    |  OpenPTXas (Python)
    v
SM_120 cubin (ELF binary)
    |  cuModuleLoad + cuLaunchKernel
    v
RTX 5090 GPU --> correct results
```

## Quick Start

```bash
# Compile a CUDA kernel to PTX
python -m opencuda kernel.cu --emit-ptx

# Compile to executable cubin (requires OpenPTXas)
python -m opencuda kernel.cu --out kernel.cubin

# Run the compiler test suite (31,000+ tests)
pytest opencuda/tests/ -x -q
```

## What's Supported

| Feature | Status |
|---------|--------|
| Arithmetic (`+ - * / %`), bitwise, compound assignment | Working |
| Control flow (if/else, for, while, do/while, switch) | Working |
| Ternary expressions (`cond ? a : b`) | Working (GPU-verified) |
| Float/int types, pointers, structs | Working |
| Shared memory + `__syncthreads()` | Working |
| Atomics (9 ops) | Working |
| Warp shuffles, ballot | Working |
| Device functions with inlining | Working |
| Multi-kernel files | Working |
| Optimization (constant folding, CSE, copy propagation, loop unrolling) | Working |

## Architecture

Pure Python. No dependencies beyond pytest.

- **Frontend**: Lexer + recursive descent parser + SSA IR
- **Optimizer**: Constant folding, CSE, copy propagation, loop unrolling
- **Codegen**: PTX 9.0 emission with ternary diamond detection, register peepholes
- **Backend**: [OpenPTXas](https://github.com/garrick99/openptxas) for PTX-to-cubin

## Requirements

- Python 3.11+
- [OpenPTXas](https://github.com/garrick99/openptxas) (for cubin generation)
- NVIDIA GPU + CUDA toolkit (for execution)

## License

See LICENSE file.
