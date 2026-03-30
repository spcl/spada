#!/bin/sh
# E2E test: 2-D allreduce (chain reduce + multicast broadcast, NX*NY PEs, K elements per PE).
# Kernel: allreduce_2D.sptl  params: NX NY K
# Reference: OUT_out[i,j,:] == sum(a_in[:,:,:])  for all i,j

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="allreduce_2d_sptl"

run_allreduce_2d() {
    NX=$1
    NY=$2
    K=$3
    echo "--- allreduce_2d NX=$NX NY=$NY K=$K ---"

    sptlc "$COLLECTIVES_DIR/allreduce_2D.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand($NX, $NY, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_allreduce_2d
    cleanup "$FOLDER"
}

run_allreduce_2d 2 2 2
run_allreduce_2d 4 4 4
run_allreduce_2d 3 4 4
run_allreduce_2d 4 3 8
