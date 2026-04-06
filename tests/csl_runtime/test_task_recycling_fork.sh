#!/bin/sh
# E2E test: greedy-coloring recycler — chain with 4-way fork/join.
# A 12-step sequential chain feeds 4 concurrent fork arms, all joined by
# awaitall.  The fork arms conflict in the coloring graph (they may be live
# simultaneously) so the recycler must place them in distinct hardware slots.
# Together with the chain the total local-task count exceeds 13 slots on WSE2.
# Expected output: scalar sum of all 16 input elements.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDER="task_recycling_fork_sptl"
RUNTIME_PY="$(cd "$SCRIPT_DIR/../.." && pwd)/spatialstencil/runtime/runtime.py"

sptlc "$SCRIPT_DIR/samples/task_recycling_fork.sptl" "$FOLDER" --disable-task-fusion

python3 - <<'PYEOF'
import numpy as np
data = np.arange(1.0, 17.0, dtype=np.float32).reshape(1, 1, 16)
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
print("Test passed: fork recycling output matches expected sum.")
PYEOF

rm -rf "$FOLDER"
rm -f input.npy OUT_output.npy
