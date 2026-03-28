#!/bin/sh
# E2E test: simple y-axis multicast.
# PE (0,0) receives one f32 from the host and multicasts it to PEs (0,1)…(0,K-1).
# Each receiver writes its copy to the output.
# Reference: OUT_out[0, k, 0] == a_in[0, 0, 0]  for k in 0..K-2.
# Tested with K ∈ {2, 3, 5, 8}.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../../samples/spatial/testing" && pwd)"
FOLDER="multicast_simple_y_sptl"

run_multicast_y() {
    K=$1
    echo "--- multicast_simple_y K=$K ---"

    sptlc "$TESTING_DIR/multicast_simple_y.sptl" "$FOLDER" -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 1).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_y $K
    cleanup "$FOLDER"
}

run_multicast_y 2
run_multicast_y 3
run_multicast_y 5
run_multicast_y 8
