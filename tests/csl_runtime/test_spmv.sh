#!/bin/sh
# E2E test: distributed sparse GEMV y = alpha * A * x + beta * y.
# Grid PX × PY; each PE holds a padded COO block of A with bound NZ.
# Phase 1: load COO A; Phase 2: load x (j=0) and y (i=0); Phase 3: broadcast x in Y;
# Phase 4: local COO SpMV; Phase 5: pipelined chain reduce z in X,
#   root applies alpha*z + beta*y and outputs result.
# Reference: OUT_out.npy[0, j, :] == (alpha * A_full @ x_flat + beta * y_flat)[j*K:(j+1)*K]
# Tested with (PX, PY) ∈ {(2,2), (2,3), (3,2), (3,4)}, K=2, NZ=K*K,
# and a sparser case (2,2) with NZ=2 < K*K and explicit zero padding.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=2
BLAS_DIR="$(cd "$(dirname "$0")/../../samples/spatial/blas" && pwd)"
FOLDER="spmv_sptl"

pack_and_run() {
    px=$1
    py=$2
    nz=$3
    n_nnz=$4
    echo "--- spmv PX=$px PY=$py K=$K NZ=$nz n_nnz=$n_nnz ---"

    sptlc "$BLAS_DIR/spmv.sptl" "$FOLDER" -p PX=$px -p PY=$py -p K=$K -p NZ=$nz

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
PX, PY, K, NZ, N_NNZ = $px, $py, $K, $nz, $n_nnz
assert 1 <= N_NNZ <= NZ
x     = np.random.rand(PX, 1, K).astype(np.float32)
y     = np.random.rand(1, PY, K).astype(np.float32)
alpha = np.array([0.5], dtype=np.float32)
beta  = np.array([2.0], dtype=np.float32)
A_val = np.zeros((PX, PY, NZ), dtype=np.float32)
A_row = np.zeros((PX, PY, NZ), dtype=np.int16)
A_col = np.zeros((PX, PY, NZ), dtype=np.int16)
for i in range(PX):
    for j in range(PY):
        coords = [(r, c) for r in range(K) for c in range(K)]
        np.random.shuffle(coords)
        for p, (r, c) in enumerate(coords[:N_NNZ]):
            A_row[i, j, p] = r
            A_col[i, j, p] = c
            A_val[i, j, p] = np.random.rand()
np.save('x.npy',     x)
np.save('A_val.npy', A_val)
np.save('A_row.npy', A_row)
np.save('A_col.npy', A_col)
np.save('y.npy',     y)
np.save('alpha.npy', alpha)
np.save('beta.npy',  beta)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" x.npy A_val.npy A_row.npy A_col.npy y.npy alpha.npy beta.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
PX, PY, K, NZ = $px, $py, $K, $nz
x_npy = np.load('x.npy')          # (PX, 1, K)
A_val = np.load('A_val.npy')      # (PX, PY, NZ)
A_row = np.load('A_row.npy')      # (PX, PY, NZ)
A_col = np.load('A_col.npy')      # (PX, PY, NZ)
y_npy = np.load('y.npy')          # (1, PY, K)
alpha = float(np.load('alpha.npy')[0])
beta  = float(np.load('beta.npy')[0])
out   = np.load('OUT_out.npy')    # (1, PY, K)
x_flat = x_npy.reshape(PX * K)
y_flat = y_npy.reshape(PY * K)
A_full = np.zeros((PY * K, PX * K), dtype=np.float32)
for i in range(PX):
    for j in range(PY):
        for p in range(NZ):
            r = int(A_row[i, j, p])
            c = int(A_col[i, j, p])
            A_full[j*K + r, i*K + c] += A_val[i, j, p]
y_ref = (alpha * (A_full @ x_flat) + beta * y_flat).astype(np.float32)
y_out = out.reshape(PY * K)
if not np.allclose(y_out, y_ref, atol=1e-4):
    print(f"FAILED PX={PX} PY={PY} NZ={NZ}: max abs diff = {float(np.max(np.abs(y_out - y_ref))):.3e}")
    print(f"  expected: {y_ref}")
    print(f"  got:      {y_out}")
    sys.exit(1)
print(f"Passed PX={PX} PY={PY} NZ={NZ}: spmv output matches alpha * A @ x + beta * y.")
PYEOF

    rm -rf "$FOLDER" x.npy A_val.npy A_row.npy A_col.npy y.npy alpha.npy beta.npy OUT_out.npy
}

# Dense-capacity COO: NZ = K*K, every entry stored (no padding slots).
NZ_DENSE=$((K * K))
pack_and_run 2 2 $NZ_DENSE $NZ_DENSE
pack_and_run 2 3 $NZ_DENSE $NZ_DENSE
pack_and_run 3 2 $NZ_DENSE $NZ_DENSE
pack_and_run 3 4 $NZ_DENSE $NZ_DENSE

# Sparse COO: NZ < K*K, actual nnz < NZ so remaining slots are (0, 0, 0.0) padding.
pack_and_run 2 2 2 1
