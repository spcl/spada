#!/bin/sh
# E2E test: distributed matrix-vector multiply y = A * x.
# Grid PX × PY; each PE holds a K×K block of A.
# Phase 1: load A; Phase 2: load x (j=0); Phase 3: broadcast x in Y;
# Phase 4: local matmul z=A@x; Phase 5: pipelined chain reduce z in X, output y.
# Reference: OUT_out.npy[0, j, :] == (A_full @ x_flat)[j*K:(j+1)*K]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

PX=3
PY=4
K=2
FOLDER="matvec_sptl"

BLAS_DIR="$(cd "$(dirname "$0")/../../samples/spatial/blas" && pwd)"

sptlc "$BLAS_DIR/matvec.sptl" "$FOLDER" -p PX=$PX -p PY=$PY -p K=$K

python3 - <<PYEOF
import numpy as np
np.random.seed(42)
x = np.random.rand($PX, 1, $K).astype(np.float32)
A = np.random.rand($PX, $PY, $K * $K).astype(np.float32)
np.save('x.npy', x)
np.save('A.npy', A)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" x.npy A.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys

PX, PY, K = 3, 4, 2

x_npy = np.load('x.npy')        # (PX, 1, K): x[i*K:(i+1)*K] at PE(i,0)
A_npy = np.load('A.npy')        # (PX, PY, K*K): row-major K×K block at PE(i,j)
out   = np.load('OUT_out.npy')  # (1, PY, K): y[j*K:(j+1)*K] at PE(0,j)

x_flat = x_npy.reshape(PX * K)

# Assemble the full (PY*K) × (PX*K) matrix from per-PE blocks.
# PE(i,j) holds A_full[j*K:(j+1)*K, i*K:(i+1)*K].
A_full = np.zeros((PY * K, PX * K), dtype=np.float32)
for i in range(PX):
    for j in range(PY):
        A_full[j*K:(j+1)*K, i*K:(i+1)*K] = A_npy[i, j, :].reshape(K, K)

y_ref = (A_full @ x_flat).astype(np.float32)
y_out = out.reshape(PY * K)

if not np.allclose(y_out, y_ref, atol=1e-4):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(y_out - y_ref))):.3e}")
    print(f"  expected: {y_ref}")
    print(f"  got:      {y_out}")
    sys.exit(1)
print("Test passed: matvec output matches A @ x.")
PYEOF

cleanup "$FOLDER"
rm -f x.npy A.npy
