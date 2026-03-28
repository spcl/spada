#!/bin/sh
# E2E test: distributed GEMV y = alpha * A * x + beta * y.
# Grid PX × PY; each PE holds a K×K block of A.
# Phase 1: load A; Phase 2: load x (j=0) and y (i=0); Phase 3: broadcast x in Y;
# Phase 4: local matmul z=A@x; Phase 5: pipelined chain reduce z in X,
#   root applies alpha*z + beta*y and outputs result.
# Reference: OUT_out.npy[0, j, :] == (alpha * A_full @ x_flat + beta * y_flat)[j*K:(j+1)*K]
# Tested with (PX, PY) ∈ {(2,2), (2,3), (3,2), (3,4)}, K=2.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=2
BLAS_DIR="$(cd "$(dirname "$0")/../../samples/spatial/blas" && pwd)"
FOLDER="gemv_sptl"

run_gemv() {
    px=$1
    py=$2
    echo "--- gemv PX=$px PY=$py K=$K ---"

    sptlc "$BLAS_DIR/gemv.sptl" "$FOLDER" -p PX=$px -p PY=$py -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
x     = np.random.rand($px, 1,   $K      ).astype(np.float32)
A     = np.random.rand($px, $py, $K * $K ).astype(np.float32)
y     = np.random.rand(1,   $py, $K      ).astype(np.float32)
alpha = np.array([0.5], dtype=np.float32)
beta  = np.array([2.0], dtype=np.float32)
np.save('x.npy',     x)
np.save('A.npy',     A)
np.save('y.npy',     y)
np.save('alpha.npy', alpha)
np.save('beta.npy',  beta)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" x.npy A.npy y.npy alpha.npy beta.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
PX, PY, K = $px, $py, $K
x_npy = np.load('x.npy')        # (PX, 1, K)
A_npy = np.load('A.npy')        # (PX, PY, K*K)
y_npy = np.load('y.npy')        # (1, PY, K)
alpha = float(np.load('alpha.npy')[0])
beta  = float(np.load('beta.npy')[0])
out   = np.load('OUT_out.npy')  # (1, PY, K)
x_flat = x_npy.reshape(PX * K)
y_flat = y_npy.reshape(PY * K)
A_full = np.zeros((PY * K, PX * K), dtype=np.float32)
for i in range(PX):
    for j in range(PY):
        A_full[j*K:(j+1)*K, i*K:(i+1)*K] = A_npy[i, j, :].reshape(K, K)
y_ref = (alpha * (A_full @ x_flat) + beta * y_flat).astype(np.float32)
y_out = out.reshape(PY * K)
if not np.allclose(y_out, y_ref, atol=1e-4):
    print(f"FAILED PX={PX} PY={PY}: max abs diff = {float(np.max(np.abs(y_out - y_ref))):.3e}")
    print(f"  expected: {y_ref}")
    print(f"  got:      {y_out}")
    sys.exit(1)
print(f"Passed PX={PX} PY={PY}: gemv output matches alpha * A @ x + beta * y.")
PYEOF

    rm -rf "$FOLDER" x.npy A.npy y.npy alpha.npy beta.npy OUT_out.npy
}

run_gemv 2 2
run_gemv 2 3
run_gemv 3 2
run_gemv 3 4
