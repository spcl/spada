#!/bin/sh
# E2E test: pipelined 2-D broadcast from PE (0,0) to all NX*NY PEs, K elements.
# Kernel: broadcast_2D.sptl  params: NX NY K
# Reference: OUT_out[i,j,:] == a_in[0,0,:]  for all i in 0..NX-1, j in 0..NY-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

NX=4
NY=4
K=4
FOLDER="broadcast_2d_sptl"

sptlc "$COLLECTIVES_DIR/broadcast_2D.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

python3 - <<PYEOF
import numpy as np
a = np.random.rand(1, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_broadcast_2d $NX $NY
cleanup "$FOLDER"
