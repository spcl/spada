#!/bin/sh
# E2E test: 2-D two-phase allreduce (two-phase reduce + multicast broadcast).
# Kernel: twophase_allreduce_2D.sptl  params: GX SX GY SY K
# Constraints: SX and SY must be even.
# Reference: OUT_out[i,j,:] == sum(a_in[:,:,:])  for all i,j

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="twophase_allreduce_2d_sptl"

run_twophase_allreduce_2d() {
    GX=$1
    SX=$2   # must be even
    GY=$3
    SY=$4   # must be even
    K=$5
    echo "--- twophase_allreduce_2d GX=$GX SX=$SX GY=$GY SY=$SY K=$K ---"

    sptlc "$COLLECTIVES_DIR/twophase_allreduce_2D.sptl" "$FOLDER" \
        -p GX=$GX -p SX=$SX -p GY=$GY -p SY=$SY -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
nx = $GX * $SX
ny = $GY * $SY
a = np.random.rand(nx, ny, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_allreduce_2d
    cleanup "$FOLDER"
}

run_twophase_allreduce_2d 2 2 2 2 2
run_twophase_allreduce_2d 3 4 3 4 4
run_twophase_allreduce_2d 2 4 3 4 4
run_twophase_allreduce_2d 3 4 2 4 8
