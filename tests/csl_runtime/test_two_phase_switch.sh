#!/bin/sh
# E2E test: two streams sharing one channel, retired by `close` (two_phase_split.sptl).
#
# PEs 0..3 form a reduce tree over a single channel: `hop1` moves data one PE west, `hop2` moves it
# two PEs west, and both are declared on channel 0. Every PE closes `hop1` before `hop2` is used, so
# the routers may take the channel over.
#
# This exercises router switch positions and the switch-advance control wavelets:
#   PE 0: both streams arrive from the east   -> one configuration, no switch
#   PE 1: sends hop1, then relays hop2        -> two positions, advanced by its own close
#   PE 2: receives hop1, then sends hop2      -> two positions, advanced by PE 3's close
#   PE 3: sends hop1 only                     -> one configuration, no switch
#
# Reference: OUT_out == sum over all four input rows.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=32
FOLDER="two_phase_switch_sptl"
SAMPLES_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$SAMPLES_DIR/two_phase_split.sptl" "$FOLDER" -p K=$K

# The channel must be reused through switch positions rather than a second color
grep -q '\.switches' "$FOLDER/layout.csl" || {
    echo "Test failed: no router switch configuration was generated."
    exit 1
}

python3 - <<PYEOF
import numpy as np
a = np.random.rand(4, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
a = np.load('a_in.npy')
ref = np.sum(a, axis=0)
out = np.load('OUT_out.npy').reshape(ref.shape)
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: channel reuse across switch positions produced the correct reduction.")
PYEOF

cleanup "$FOLDER"
