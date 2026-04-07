#!/bin/sh
# E2E regression: local task IDs must not overlap with allocated stream colors.
#
# This uses only 2 simulated PEs but allocates channels/colors 0..8, which is
# enough to hit the historical collision with local task ID 8.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_PY="$(cd "$SCRIPT_DIR/../.." && pwd)/spatialstencil/runtime/runtime.py"
FOLDER="task_color_overlap_many_channels_sptl"

sptlc "$SCRIPT_DIR/samples/task_color_overlap_many_channels.sptl" "$FOLDER"

python3 - <<'PYEOF'
import numpy as np

inp = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
np.save("inp.npy", inp)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

python3 - <<'PYEOF'
import numpy as np
import sys

inp = np.load("inp.npy")
out = np.load("OUT_out.npy")
expected = np.array([[[inp[0, 0, 0] + 9.0 * inp[1, 0, 0]]]], dtype=np.float32)

if not np.allclose(out, expected, atol=1e-5):
    print(f"Test failed: expected {expected.flatten()}, got {out.flatten()}")
    sys.exit(1)

print("Test passed: many-channel color/task-ID overlap regression is covered.")
PYEOF

rm -rf "$FOLDER"
rm -f inp.npy OUT_out.npy
