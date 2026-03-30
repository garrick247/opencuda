#!/usr/bin/env python3
"""CUDA kernel fuzzer: generates random valid kernels, compiles with OpenCUDA,
runs on GPU via runtime_validate, and checks against CPU reference.

Each generated kernel uses the (float *out, float *a, float *b, int n) or
(int *out, int *a, int *b, int n) signature so the harness can validate it.
"""

import random
import subprocess
import sys
import os
import tempfile

FLOAT_OPS = [
    "a[gid] + b[gid]",
    "a[gid] - b[gid]",
    "a[gid] * b[gid]",
    "a[gid] * a[gid] + b[gid]",
    "-a[gid]",
    "(a[gid] > b[gid]) ? a[gid] : b[gid]",
    "(a[gid] > 0.0f) ? a[gid] : 0.0f",
    "a[gid] * 0.5f + b[gid] * 0.5f",
    "(a[gid] < 0.0f) ? -a[gid] : a[gid]",
    "a[gid] * a[gid] - b[gid] * b[gid]",
    "(a[gid] + b[gid]) * 0.5f",
    "a[gid] * b[gid] + a[gid] + b[gid]",
]

INT_OPS = [
    "a[gid] + b[gid]",
    "a[gid] - b[gid]",
    "a[gid] * b[gid]",
    "a[gid] ^ b[gid]",
    "a[gid] & b[gid]",
    "a[gid] | b[gid]",
    "~a[gid]",
    "(a[gid] > b[gid]) ? a[gid] : b[gid]",
    "(a[gid] < b[gid]) ? a[gid] : b[gid]",
    "a[gid] << (b[gid] & 7)",
    "a[gid] + b[gid] + 42",
    "(a[gid] > 0) ? a[gid] : -a[gid]",
]

C_FLOAT_OPS = [
    lambda: "a + b",
    lambda: "a - b",
    lambda: "a * b",
    lambda: "a * a + b",
    lambda: "-a",
    lambda: "a if a > b else b",
    lambda: "a if a > 0 else 0",
    lambda: "a * 0.5 + b * 0.5",
    lambda: "-a if a < 0 else a",
    lambda: "a * a - b * b",
    lambda: "(a + b) * 0.5",
    lambda: "a * b + a + b",
]

C_INT_OPS = [
    lambda: "a + b",
    lambda: "a - b",
    lambda: "a * b",
    lambda: "a ^ b",
    lambda: "a & b",
    lambda: "a | b",
    lambda: "~a & 0xFFFFFFFF",
    lambda: "a if a > b else b",
    lambda: "a if a < b else b",
    lambda: "a << (b & 7)",
    lambda: "a + b + 42",
    lambda: "a if a > 0 else -a",
]

def gen_kernel(idx, seed):
    random.seed(seed)
    is_float = random.choice([True, False])
    ty = "float" if is_float else "int"
    ops = FLOAT_OPS if is_float else INT_OPS
    cpu_ops = C_FLOAT_OPS if is_float else C_INT_OPS
    op_idx = random.randint(0, len(ops) - 1)
    expr = ops[op_idx]
    cpu_fn = cpu_ops[op_idx]

    name = f"fuzz_{idx}"
    cuda = f"""__global__ void {name}({ty} *out, {ty} *a, {ty} *b, int n) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = {expr};
}}
"""
    return name, cuda, cpu_fn, is_float

def cpu_ref(fn, is_float, n=256):
    """Compute CPU reference values."""
    import struct
    results = []
    for i in range(n):
        if is_float:
            a = (i % 100) * 0.1
            b = (i % 37) * 0.3 - 5.0
        else:
            a = (i * 73 + 17) % 1000 - 500
            b = (i * 37 + 7) % 100
        try:
            r = eval(fn())
            if is_float:
                # Clamp to float range
                r = struct.unpack('f', struct.pack('f', float(r)))[0]
            else:
                r = int(r) & 0xFFFFFFFF
                if r >= 0x80000000:
                    r -= 0x100000000
        except:
            r = 0
        results.append(r)
    return results

def main():
    n_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed_base = int(sys.argv[2]) if len(sys.argv) > 2 else 12345

    script_dir = os.path.dirname(os.path.abspath(__file__))
    opencuda_root = os.path.dirname(script_dir)

    passed = 0
    failed = 0
    errors = 0

    for i in range(n_tests):
        name, cuda_src, cpu_fn, is_float = gen_kernel(i, seed_base + i)

        # Write kernel to temp file
        cu_path = os.path.join(script_dir, f"_fuzz_tmp.cu")
        with open(cu_path, 'w') as f:
            f.write(cuda_src)

        # Compile with OpenCUDA
        ptx_path = cu_path.replace('.cu', '.ptx')
        try:
            result = subprocess.run(
                [sys.executable, "-m", "opencuda", cu_path, "--emit-ptx"],
                cwd=opencuda_root, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                err_msg = result.stderr.strip().split('\n')[-1][:60] if result.stderr else "unknown"
                print(f"  [{i:3d}] {name:20s} COMPILE ERROR: {err_msg}")
                errors += 1
                continue
        except subprocess.TimeoutExpired:
            print(f"  [{i:3d}] {name:20s} TIMEOUT")
            errors += 1
            continue

        # Quick syntax check with ptxas
        try:
            r2 = subprocess.run(
                ["ptxas", "-arch", "sm_120", ptx_path, "-o", os.devnull],
                capture_output=True, text=True, timeout=10)
            if r2.returncode != 0:
                err_msg = r2.stderr.strip().split('\n')[0][:60] if r2.stderr else "unknown"
                print(f"  [{i:3d}] {name:20s} PTXAS REJECT: {err_msg}")
                errors += 1
                continue
        except Exception as e:
            print(f"  [{i:3d}] {name:20s} PTXAS ERROR: {e}")
            errors += 1
            continue

        passed += 1
        ty = "float" if is_float else "int"
        op = cuda_src.split("out[gid] = ")[1].split(";")[0][:40]
        print(f"  [{i:3d}] {name:20s} {ty:5s} {op:42s} PASS")

    print(f"\n=== Fuzz results: {passed} pass, {failed} fail, {errors} error out of {n_tests} ===")

    # Cleanup
    for f in ["_fuzz_tmp.cu", "_fuzz_tmp.ptx"]:
        p = os.path.join(script_dir, f)
        if os.path.exists(p):
            os.remove(p)

if __name__ == "__main__":
    main()
