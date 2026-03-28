#!/bin/sh
# E2E test: binary-tree 1-D reduction (2^L PEs, K elements per PE).
# Kernel: tree_reduce_1D.sptl  params: L K
# Reference: OUT_out[0,0,:] == sum(a_in[:, 0, :], axis=0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

L=2   # 2^L = 4 PEs
K=4
FOLDER="tree_reduce_1d_sptl"

sptlc "$COLLECTIVES_DIR/tree_reduce_1D.sptl" "$FOLDER" -p L=$L -p K=$K

python3 - <<PYEOF
import numpy as np
n_pes = 1 << $L   # 2^L
a = np.random.rand(n_pes, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum
cleanup "$FOLDER"
