#!/bin/sh
# E2E test: a bounded stream closing itself (bounded_chain.sptl).
#
# `stream<f32, K> eastwards` carries exactly K elements per stream edge, so it closes itself without
# any explicit `close` in the source. That implicit close is what advances the middle PE's router
# from its receiving configuration to its sending one -- both directions share channel 0.
#
# Reference: OUT_out == inp[0] + inp[1] (the third input row is never read).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=4
FOLDER="bounded_chain_sptl"
SAMPLES_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$SAMPLES_DIR/bounded_chain.sptl" "$FOLDER" -p K=$K

# The forwarding PE needs two configurations, and the head PE retires the first one for it
grep -q '\.switches' "$FOLDER/layout.csl" || {
    echo "Test failed: no router switch configuration was generated."
    exit 1
}
grep -q 'SWITCH_ADV' "$FOLDER"/code_0_0.csl || {
    echo "Test failed: the bounded stream did not emit a switch advance."
    exit 1
}

python3 - <<PYEOF
import numpy as np
a = np.random.rand(3, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
a = np.load('a_in.npy')
ref = a[0] + a[1]
out = np.load('OUT_out.npy').reshape(ref.shape)
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()}")
    print(f"  got:      {out.flatten()}")
    sys.exit(1)
print("Test passed: bounded stream closed itself and the chain produced the correct sum.")
PYEOF

cleanup "$FOLDER"
