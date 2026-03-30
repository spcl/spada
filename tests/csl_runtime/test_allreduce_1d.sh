#!/bin/sh
# E2E test: 1-D allreduce (chain reduce + multicast broadcast, N PEs, K elements per PE).
# Kernel: allreduce_1D.sptl  params: N K
# Reference: OUT_out[i,0,:] == sum(a_in[:,0,:])  for all i in 0..N-1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

FOLDER="allreduce_1d_sptl"

run_allreduce_1d() {
    N=$1
    K=$2
    echo "--- allreduce_1d N=$N K=$K ---"

    sptlc "$COLLECTIVES_DIR/allreduce_1D.sptl" "$FOLDER" -p N=$N -p K=$K

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
a = np.random.rand($N, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_allreduce
    cleanup "$FOLDER"
}

run_allreduce_1d 2 2
run_allreduce_1d 4 4
run_allreduce_1d 8 4
run_allreduce_1d 4 8
