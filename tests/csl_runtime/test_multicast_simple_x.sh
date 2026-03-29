#!/bin/sh
# E2E test: simple x-axis multicast.
# PE (0,0) receives one f32 from the host and multicasts it to PEs (1,0)…(K-1,0).
# Each PE writes its copy to the output.
# Reference: OUT_out[k, 0, 0] == a_in[0, 0, 0]  for k in 0..K-1.
# Tested with K ∈ {2, 3, 5, 8}.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../../samples/spatial/testing" && pwd)"
FOLDER="multicast_simple_x_sptl"

run_multicast_x() {
    K=$1
    echo "--- multicast_simple_x K=$K ---"

    sptlc "$TESTING_DIR/multicast_simple_x.sptl" "$FOLDER" -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_x $K
    cleanup "$FOLDER"
}

run_multicast_x 2
run_multicast_x 3
run_multicast_x 5
run_multicast_x 8
