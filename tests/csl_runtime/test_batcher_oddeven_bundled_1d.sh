#!/bin/sh
# E2E test: the bundled 1D Batcher odd-even mergesort (2^L PEs, one f32 key per PE).
# Kernel: batcher_oddeven_bundled_1D.sptl  params: L
# Same result as batcher_oddeven_1D, but each phase runs on two colors instead of one per
# comparator: OUT_out[:, 0, 0] == sort(inp[:, 0, 0]).
# L <= 3, which is where the three filters a PE can use run out.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SORT_DIR="$(cd "$(dirname "$0")/../../samples/spatial/sort" && pwd)"
FOLDER="batcher_oddeven_bundled_1d_sptl"

run_batcher() {
    l=$1
    echo "--- batcher_oddeven_bundled_1d L=$l ---"

    sptlc "$SORT_DIR/batcher_oddeven_bundled_1D.sptl" "$FOLDER" -p L=$l

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
n = 1 << $l
inp = np.random.rand(n, 1, 1).astype(np.float32)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 240 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
n = 1 << $l
inp = np.load('inp.npy').reshape(n)
out = np.load('OUT_out.npy').reshape(n)
ref = np.sort(inp)
if not np.allclose(out, ref, atol=1e-6):
    print(f"FAILED L=$l: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print(f"Passed L=$l: output matches sorted input.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_batcher 1
run_batcher 2
run_batcher 3
