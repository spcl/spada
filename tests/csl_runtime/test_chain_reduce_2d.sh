#!/bin/sh
# E2E test: pipelined 2-D chain reduction (NX*NY PEs, K elements per PE).
# Kernel: chain_reduce_2D.sptl  params: NX NY K
# Reference: OUT_out[0,0,:] == sum(a_in[:,:,:])

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

NX=4
NY=4
K=4
FOLDER="chain_reduce_2d_sptl"

sptlc "$COLLECTIVES_DIR/chain_reduce_2D.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

python3 - <<PYEOF
import numpy as np
a = np.random.rand($NX, $NY, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum_2d
cleanup "$FOLDER"
