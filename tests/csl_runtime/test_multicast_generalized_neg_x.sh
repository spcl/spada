#!/bin/sh
# E2E test: generalized x-axis negative-direction multicast.
#
# Sender at PE (RE-1, 0) receives one f32 from the host, writes its copy to
# out[RE-1, 0], and multicasts WEST via range [-RS:-RE] to PEs (0,0)…(RE-RS-1,0).
# PEs (RE-RS,0)…(RE-2,0) are relay-only when RS > 1.
#
# Reference: OUT_out[RE-1, 0, 0] == a_in[0, 0, 0]  (sender)
#            OUT_out[k, 0, 0]    == a_in[0, 0, 0]  for k in 0..RE-RS-1 (receivers)
#
# Tested with (RS, RE): (1,2), (1,3), (1,5), (2,5).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

TESTING_DIR="$(cd "$(dirname "$0")/../../samples/spatial/testing" && pwd)"
FOLDER="multicast_generalized_neg_x_sptl"

run_multicast_x_neg() {
    RS=$1
    RE=$2
    echo "--- multicast_generalized_x_neg RS=$RS RE=$RE ---"

    sptlc "$TESTING_DIR/multicast_generalized_x_neg.sptl" "$FOLDER" -p RS=$RS -p RE=$RE

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand(1, 1, 2).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 60 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_multicast_x_neg $RS $RE
    cleanup "$FOLDER"
}

run_multicast_x_neg 2 5
run_multicast_x_neg 3 8