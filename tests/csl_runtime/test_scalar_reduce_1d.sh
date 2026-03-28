#!/bin/sh
# E2E test: scalar 1-D chain reduction (N PEs, K=1 element per PE).
# Kernel: scalar_reduce_1D.sptl  params: N
# Reference: OUT_out[0,0,0] == sum(a_in[:, 0, 0])

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

N=4
FOLDER="scalar_reduce_1d_sptl"

sptlc "$COLLECTIVES_DIR/scalar_reduce_1D.sptl" "$FOLDER" -p N=$N

python3 - <<PYEOF
import numpy as np
a = np.random.rand($N, 1, 1).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

verify_reduce_sum
cleanup "$FOLDER"
