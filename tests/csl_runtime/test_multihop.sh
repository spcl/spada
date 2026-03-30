#!/bin/sh
# E2E test: 2-hop westward relay (multihop.sptl).
# PE 2 sends K f32 values to PE 0 via PE 1 (pure routing pass-through).
# Reference: OUT_out == inp (passthrough — data is forwarded unchanged).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=4
FOLDER="multihop_sptl"
SAMPLES_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$SAMPLES_DIR/multihop.sptl" "$FOLDER" -p K=$K

python3 - <<PYEOF
import numpy as np
a = np.random.rand(1, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
if not np.allclose(out, inp, atol=1e-6):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - inp))):.3e}")
    print(f"  expected: {inp.flatten()}")
    print(f"  got:      {out.flatten()}")
    sys.exit(1)
print("Test passed: multi-hop relay output matches input.")
PYEOF

cleanup "$FOLDER"
