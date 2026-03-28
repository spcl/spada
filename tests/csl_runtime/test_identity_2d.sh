#!/bin/sh
# E2E test: 2D identity copy (out[i,j] == inp[i,j]).
# Catches mismatched host↔device memcpy order when both grid dimensions > 1.
# Each PE gets a unique value; any PE-swap shows up as a mismatch.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

NX=3
NY=4
K=2
FOLDER="identity_2d_sptl"
TESTING_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/testing" && pwd)"

sptlc "$TESTING_DIR/identity_2d.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

python3 - <<PYEOF
import numpy as np
np.random.seed(7)
a = np.random.rand($NX, $NY, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<PYEOF
import numpy as np, sys
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
if not np.allclose(out, inp, atol=1e-6):
    print(f"FAILED: max abs diff = {float(np.max(np.abs(out - inp))):.3e}")
    print(f"  expected: {inp.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: 2D identity output matches input.")
PYEOF

rm -rf "$FOLDER"
rm -f a_in.npy OUT_out.npy
