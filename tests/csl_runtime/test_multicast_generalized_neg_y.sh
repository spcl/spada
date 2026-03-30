#!/bin/sh
# E2E test: generalized y-axis negative-direction multicast.
#
# Sender at PE (0, RE-1) receives one f32 from the host, writes its copy to
# out[0, RE-1], and multicasts NORTH via range [-RS:-RE] to PEs (0,0)…(0,RE-RS-1).
# PEs (0,RE-RS)…(0,RE-2) are relay-only when RS > 1.
#
# Reference: OUT_out[0, RE-1, 0] == a_in[0, 0, 0]  (sender)
#            OUT_out[0, k, 0]    == a_in[0, 0, 0]  for k in 0..RE-RS-1 (receivers)
#
# Tested with (RS, RE): (1,2), (1,3), (1,5), (2,5).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../../samples/spatial/testing" && pwd)"
FOLDER="multicast_generalized_neg_y_sptl"

run_multicast_y_neg() {
    RS=$1
    RE=$2
    echo "--- multicast_generalized_y_neg RS=$RS RE=$RE ---"

    sptlc "$TESTING_DIR/multicast_generalized_y_neg.sptl" "$FOLDER" -p RS=$RS RE=$RE

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_y_neg $RS $RE
    cleanup "$FOLDER"
}

run_multicast_y_neg 2 5
run_multicast_y_neg 3 8
