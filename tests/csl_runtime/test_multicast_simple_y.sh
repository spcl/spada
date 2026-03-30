#!/bin/sh
# E2E test: y-axis multicast using the generalized sample.
# PE (0,0) multicasts to PEs (0,START)…(0,N-1); PEs (0,1)…(0,START-1) are relay-only.
# Only receivers write to out (shape 1×(N-START)×2).
# Reference: OUT_out[0,k,0] == a_in[0,0,0] for k in 0..N-START-1.
#
# Tested with START=1: N∈{2,3,5,8}  (no relay PEs)
#         and START=2, N=5           (exercises gap relay PE at j=1)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../spatial_ir/samples" && pwd)"
FOLDER="multicast_simple_y_sptl"

run_multicast_y_range() {
    START=$1
    N=$2
    echo "--- multicast_generalized_y START=$START N=$N ---"

    sptlc "$TESTING_DIR/multicast_generalized_y.sptl" "$FOLDER" -p START=$START -p N=$N

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_y_range $START $N
    cleanup "$FOLDER"
}

run_multicast_y_range 1 2
run_multicast_y_range 1 3
run_multicast_y_range 2 5
run_multicast_y_range 3 5
