#!/bin/sh
# E2E test: greedy-coloring recycler — three sequential awaitall groups.
#
# Three groups of 6 completions each (18 total), each separated by an awaitall.
# Each group creates a 5-clique of blocked join tasks in the conflict graph.
# All three cliques are mutually non-conflicting (fully sequential), so they
# reuse the same 5 hardware slots, yielding ~18 local tasks in 5 slots
# (~3.5 tasks/slot).
#
# This stress-tests deeper slot reuse: 3 independent cliques of size 5 are each
# assigned one color class and recycled across three pipeline stages.
# Expected output: scalar sum of all 18 input elements.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDER="task_recycling_three_stage_sptl"
RUNTIME_PY="$(cd "$SCRIPT_DIR/../.." && pwd)/spada/runtime/runtime.py"

sptlc "$SCRIPT_DIR/samples/task_recycling_three_stage.sptl" "$FOLDER" --disable-task-fusion

python3 - <<'PYEOF'
import numpy as np
data = np.arange(1.0, 19.0, dtype=np.float32).reshape(1, 1, 18)
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
print("Test passed: three-stage recycling output matches expected sum.")
PYEOF

rm -rf "$FOLDER"
rm -f input.npy OUT_output.npy
