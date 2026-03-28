#!/bin/sh
# E2E test: pipelined 2-D two-phase reduction ((GX*SX)*(GY*SY) PEs, K elements per PE).
# Kernel: twophase_reduce_2D.sptl  params: GX SX GY SY K
# Constraints: SX and SY must be even.
# Reference: OUT_out[0,0,:] == sum(a_in[:,:,:])

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

GX=3
SX=4   # must be even
GY=3
SY=4   # must be even
K=4
FOLDER="twophase_reduce_2d_sptl"

sptlc "$COLLECTIVES_DIR/twophase_reduce_2D.sptl" "$FOLDER" \
    -p GX=$GX -p SX=$SX -p GY=$GY -p SY=$SY -p K=$K

python3 - <<PYEOF
import numpy as np
nx = $GX * $SX
ny = $GY * $SY
a = np.random.rand(nx, ny, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum_2d
cleanup "$FOLDER"
