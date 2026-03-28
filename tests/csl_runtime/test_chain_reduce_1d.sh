#!/bin/sh
# E2E test: pipelined 1-D chain reduction (N PEs, K elements per PE).
# Kernel: chain_reduce_1D.sptl  params: N K
# Reference: OUT_out[0,0,:] == sum(a_in[:, 0, :], axis=0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

N=4
K=4
FOLDER="chain_reduce_1d_sptl"

sptlc "$COLLECTIVES_DIR/chain_reduce_1D.sptl" "$FOLDER" -p N=$N -p K=$K

python3 - <<PYEOF
import numpy as np
a = np.random.rand($N, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum
cleanup "$FOLDER"
