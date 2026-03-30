#!/bin/sh
# E2E test: 1-D two-phase allreduce (two-phase reduce + multicast broadcast).
# Kernel: twophase_allreduce_1D.sptl  params: G S K
# Constraints: S must be even.
# Reference: OUT_out[i,0,:] == sum(a_in[:,0,:])  for all i in 0..G*S-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="twophase_allreduce_1d_sptl"

run_twophase_allreduce_1d() {
    G=$1
    S=$2   # must be even
    K=$3
    echo "--- twophase_allreduce_1d G=$G S=$S K=$K ---"

    sptlc "$COLLECTIVES_DIR/twophase_allreduce_1D.sptl" "$FOLDER" -p G=$G -p S=$S -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
n_pes = $G * $S
a = np.random.rand(n_pes, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_allreduce
    cleanup "$FOLDER"
}

run_twophase_allreduce_1d 2 2 2
run_twophase_allreduce_1d 3 4 4
run_twophase_allreduce_1d 4 4 4
run_twophase_allreduce_1d 3 4 8
