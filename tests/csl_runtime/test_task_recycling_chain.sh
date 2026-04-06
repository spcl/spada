#!/bin/sh
# E2E test: greedy-coloring recycler — long linear chain.
# 20 sequential async completions exceed the 13 hardware task-ID slots on WSE2,
# forcing the state-machine recycler to reuse slots across the chain.
# Expected output: scalar sum of all 20 input elements.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDER="task_recycling_chain_sptl"
RUNTIME_PY="$(cd "$SCRIPT_DIR/../.." && pwd)/spatialstencil/runtime/runtime.py"

sptlc "$SCRIPT_DIR/samples/task_recycling_chain.sptl" "$FOLDER" --disable-task-fusion

python3 - <<'PYEOF'
import numpy as np
data = np.arange(1.0, 21.0, dtype=np.float32).reshape(1, 1, 20)
np.save('input.npy', data)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" input.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
data = np.load('input.npy')
ref  = np.sum(data, axis=2, keepdims=True)
out  = np.load('OUT_output.npy')
if not np.allclose(out, ref, atol=1e-4):
    print(f"Test failed: expected {ref.flatten()}, got {out.flatten()}")
    sys.exit(1)
print("Test passed: chain recycling output matches expected sum.")
PYEOF

rm -rf "$FOLDER"
rm -f input.npy OUT_output.npy
