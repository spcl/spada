#!/bin/sh
# E2E test: pipelined 2-D binary-tree reduction ((2^LX)*(2^LY) PEs, K elements per PE).
# Kernel: tree_reduce_2D.sptl  params: LX LY K
# Reference: OUT_out[0,0,:] == sum(a_in[:,:,:])

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

LX=2   # 2^LX = 4 PEs in X
LY=2   # 2^LY = 4 PEs in Y
K=4
FOLDER="tree_reduce_2d_sptl"

sptlc "$COLLECTIVES_DIR/tree_reduce_2D.sptl" "$FOLDER" -p LX=$LX -p LY=$LY -p K=$K

python3 - <<PYEOF
import numpy as np
nx = 1 << $LX   # 2^LX
ny = 1 << $LY   # 2^LY
a = np.random.rand(nx, ny, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum_2d
cleanup "$FOLDER"
