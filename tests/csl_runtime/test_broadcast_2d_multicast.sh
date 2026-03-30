#!/bin/sh
# E2E test: 2D broadcast using direct multicast streams (X-then-Y).
# Kernel: broadcast_2D_multicast.sptl  params: NX NY K
# Reference: OUT_out[i, j, :] == a_in[0, 0, :]  for all i in 0..NX-1, j in 0..NY-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="broadcast_2d_multicast_sptl"

run_broadcast_2d_mc() {
    NX=$1
    NY=$2
    K=$3
    echo "--- broadcast_2d_multicast NX=$NX NY=$NY K=$K ---"

    sptlc "$COLLECTIVES_DIR/broadcast_2D_multicast.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_broadcast_2d $NX $NY
    cleanup "$FOLDER"
}

run_broadcast_2d_mc 2 2 2
run_broadcast_2d_mc 4 4 2
run_broadcast_2d_mc 2 3 4