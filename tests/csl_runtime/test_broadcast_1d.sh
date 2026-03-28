#!/bin/sh
# E2E test: 1-D broadcast from PE 0 to all N PEs, K elements.
# Kernel: broadcast_1D.sptl  params: N K
# Reference: OUT_out[i, 0, :] == a_in[0, 0, :]  for all i in 0..N-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

N=4
K=4
FOLDER="broadcast_1d_sptl"

sptlc "$COLLECTIVES_DIR/broadcast_1D.sptl" "$FOLDER" -p N=$N -p K=$K

python3 - <<PYEOF
import numpy as np
a = np.random.rand(1, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_broadcast $N
cleanup "$FOLDER"
