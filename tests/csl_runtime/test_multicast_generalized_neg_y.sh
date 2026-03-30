#!/bin/sh
# E2E test: generalized y-axis negative-direction multicast.
# Sender at PE (0, N-1) multicasts NORTH to PEs (0,0)…(0,N-START-1).
# PEs (0,N-START)…(0,N-2) are relay-only when START > 1.
# Only receivers write to out (shape 1×(N-START)×2); sender excluded.
# Reference: OUT_out[0,k,0] == value  for k in 0..N-START-1.
#
# Tested with START=1: N∈{2,3,5}  (no relay PEs)
#         and START=2, N=5         (exercises gap relay PE)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../spatial_ir/samples" && pwd)"
FOLDER="multicast_generalized_neg_y_sptl"

run_multicast_y_neg() {
    START=$1
    N=$2
    echo "--- multicast_generalized_y_neg START=$START N=$N ---"

    sptlc "$TESTING_DIR/multicast_generalized_y_neg.sptl" "$FOLDER" -p START=$START -p N=$N

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_y_neg $START $N
    cleanup "$FOLDER"
}

run_multicast_y_neg 1 2
run_multicast_y_neg 1 4
run_multicast_y_neg 2 5