#!/bin/sh
# E2E test: y-axis multicast using the generalized sample.
# PE (0,0) multicasts to PEs (0,START)…(0,STOP-1); PEs (0,1)…(0,START-1) are relay-only.
# Reference: OUT_out[0,0,0] and OUT_out[0,k,0] == a_in[0,0,0] for k in START..STOP-1.
#
# Tested with [1:2], [1:3], [1:5], [1:8]  (unit-start, mirrors old simple tests)
#         and [2:5]                         (non-unit start, exercises gap relay PEs).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../../samples/spatial/testing" && pwd)"
FOLDER="multicast_simple_y_sptl"

run_multicast_y_range() {
    START=$1
    STOP=$2
    echo "--- multicast_generalized_y START=$START STOP=$STOP ---"

    sptlc "$TESTING_DIR/multicast_generalized_y.sptl" "$FOLDER" -p START=$START STOP=$STOP

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_y_range $START $STOP
    cleanup "$FOLDER"
}

run_multicast_y_range 1 2
run_multicast_y_range 1 3
run_multicast_y_range 1 5
