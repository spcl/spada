#!/bin/sh
# E2E test: two-phase 1-D reduction (G groups of S PEs, K elements per PE).
# Kernel: twophase_reduce_1D.sptl  params: G S K
# Reference: OUT_out[0,0,:] == sum(a_in[:, 0, :], axis=0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

G=3
S=4   # must be even
K=4
FOLDER="twophase_reduce_1d_sptl"

run_twophase_reduce_1d() {
    EXTRA_ARGS=$@
    echo "--- twophase_reduce_1d G=$G S=$S K=$K $EXTRA_ARGS ---"
    sptlc "$COLLECTIVES_DIR/twophase_reduce_1D.sptl" "$FOLDER" -p G=$G -p S=$S -p K=$K $EXTRA_ARGS

    python3 - <<PYEOF
import numpy as np
n_pes = $G * $S
a = np.random.rand(n_pes, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    verify_reduce_sum
    cleanup "$FOLDER"
}

run_twophase_reduce_1d
run_twophase_reduce_1d --disable-copy-elision
run_twophase_reduce_1d --disable-task-recycling
run_twophase_reduce_1d --disable-task-fusion
