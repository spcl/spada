#!/bin/sh
# E2E: one channel carrying two epochs between the same pair of PEs, and nothing else.
# Kernel: samples/data_task_two_epochs.sptl  params: R (repeats of the pair of epochs).
# PE1 receives twice on channel 0, so both receives share the data task the channel binds.
# After the run both PEs hold PE0's two keys.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLE="$SCRIPT_DIR/samples/data_task_two_epochs.sptl"
FOLDER="data_task_two_epochs_sptl"

run_epochs() {
    r=$1
    echo "--- data_task_two_epochs R=$r ---"

    sptlc "$SAMPLE" "$FOLDER" -p R=$r

    python3 - <<PYEOF
import numpy as np
inp = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
out = np.load('OUT_out.npy').reshape(2, 2)
ref = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
if not np.allclose(out, ref, atol=1e-6):
    print("FAILED R=$r")
    print(f"  expected: {ref.tolist()}")
    print(f"  got:      {out.tolist()}")
    sys.exit(1)
print("Passed R=$r: both epochs arrived on the shared channel.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_epochs 1
run_epochs 3
