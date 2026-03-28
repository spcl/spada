#!/bin/sh
# E2E test: BLAS-1 axpy — y <- alpha * x + y (N×N PEs, one element each).
# Kernel: axpy.sptl  params: N
# Reference: OUT_out == alpha * x + y

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

N=8
K=4
FOLDER="axpy_sptl"
BLAS_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/blas" && pwd)"

sptlc "$BLAS_DIR/axpy.sptl" "$FOLDER" -p N=$N -p K=$K

python3 - <<PYEOF
import numpy as np
x     = np.random.rand($N, $N, $K).astype(np.float32)
alpha = np.random.rand(1).astype(np.float32)
y     = np.random.rand($N, $N, $K).astype(np.float32)
np.save('x_in.npy',     x)
np.save('alpha_in.npy', alpha)
np.save('y_in.npy',     y)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" x_in.npy alpha_in.npy y_in.npy --benchmark

python3 - <<PYEOF
import numpy as np, sys
x     = np.load('x_in.npy')
alpha = np.load('alpha_in.npy')[0]
y     = np.load('y_in.npy')
ref   = alpha * x + y
out   = np.load('OUT_out.npy')
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: axpy output matches expected alpha * x + y.")
PYEOF

rm -rf "$FOLDER"
rm -f x_in.npy alpha_in.npy y_in.npy OUT_out.npy
