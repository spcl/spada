#!/bin/sh
# E2E test: 1D scalar chain reduction over a single channel (scalar_reduce_1D.sptl).
#
# Every PE but the last receives a partial sum from the east and sends the accumulated value west,
# both on channel 0. Receiving and sending need incompatible router configurations, so each middle
# PE holds two switch positions and is advanced onto the second by the control wavelet that retires
# its upstream neighbour's stream. This is the sample that could not be lowered before router
# switching existed.
#
# Reference: OUT_out == sum over all input rows.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

N=4
FOLDER="scalar_reduce_1d_sptl"

sptlc "$COLLECTIVES_DIR/scalar_reduce_1D.sptl" "$FOLDER" -p N=$N

grep -q '\.switches' "$FOLDER/layout.csl" || {
    echo "Test failed: no router switch configuration was generated."
    exit 1
}

python3 - <<PYEOF
import numpy as np
a = np.random.rand($N, 1).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
a = np.load('a_in.npy')
ref = np.sum(a, axis=0)
out = np.load('OUT_out.npy').reshape(ref.shape)
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: expected {ref.flatten()}, got {out.flatten()}")
    sys.exit(1)
print("Test passed: chain reduction over a switched channel matches the expected sum.")
PYEOF

cleanup "$FOLDER"
