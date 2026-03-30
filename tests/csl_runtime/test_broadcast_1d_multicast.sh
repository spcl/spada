#!/bin/sh
# E2E test: 1D broadcast using direct multicast stream.
# Kernel: broadcast_1D_multicast.sptl  params: N K
# Reference: OUT_out[i, 0, :] == a_in[0, 0, :]  for all i in 0..N-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="broadcast_1d_multicast_sptl"

run_broadcast_1d_mc() {
    N=$1
    K=$2
    echo "--- broadcast_1d_multicast N=$N K=$K ---"

    sptlc "$COLLECTIVES_DIR/broadcast_1D_multicast.sptl" "$FOLDER" -p N=$N -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_broadcast $N
    cleanup "$FOLDER"
}

run_broadcast_1d_mc 2 2
run_broadcast_1d_mc 4 2
run_broadcast_1d_mc 8 4
run_broadcast_1d_mc 4 8
