#!/bin/sh
# E2E test: greedy-coloring recycler — two sequential awaitall groups.
#
# Two groups of 8 completions each (16 total), separated by an awaitall.
# Each group creates a 7-clique of blocked join tasks in the conflict graph.
# The two cliques are fully sequential so they share the same 7 hardware slots,
# producing ~15 local tasks mapped into 7 slots (~2 tasks/slot).
#
# This exercises genuine slot reuse across non-conflicting stages, which is the
# main property added by the load-balanced greedy coloring.
# Expected output: scalar sum of all 16 input elements.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDER="task_recycling_two_stage_sptl"
RUNTIME_PY="$(cd "$SCRIPT_DIR/../.." && pwd)/spatialstencil/runtime/runtime.py"

sptlc "$SCRIPT_DIR/samples/task_recycling_two_stage.sptl" "$FOLDER" --disable-task-fusion

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
print("Test passed: two-stage recycling output matches expected sum.")
PYEOF

rm -rf "$FOLDER"
rm -f input.npy OUT_output.npy
