#!/bin/sh
# E2E test: the WSE-3 1D Batcher odd-even mergesort (2^L PEs, a block of K f32 keys per PE).
# Kernel: batcher_oddeven_wse3_1D.sptl  params: L, K
# Same result as batcher_oddeven_1D -- OUT_out.reshape(n*k) == sort(inp.reshape(n*k)) -- and the same
# network as the bundled variant. What it adds is L = 4, which the bundled variant cannot reach on
# wse3: there a queue is bound to a color for the whole kernel, so a PE needs one per color it ever
# uses, and the bundled variant wants seven of the six. Pooling the unbundled phases by the origin's
# residue rather than by direction spends one queue per distance in each direction instead of two,
# which brings it to five. The routers switch for it, on twelve colors of the fifteen wse3 can switch.
# This kernel is written for the queue model of wse3 and is only run there; the bundled variant is
# the better fit on wse2, whose routers here would not switch at all.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SORT_DIR="$(cd "$(dirname "$0")/../../samples/spatial/sort" && pwd)"
FOLDER="batcher_oddeven_wse3_1d_sptl"

if [ "${WSE_ARCH:-wse2}" != "wse3" ]; then
    echo "Skipping: this variant targets wse3; on wse2 use test_batcher_oddeven_bundled_1d.sh."
    exit 0
fi

run_batcher() {
    l=$1
    k=$2
    echo "--- batcher_oddeven_wse3_1d L=$l K=$k ---"

    sptlc "$SORT_DIR/batcher_oddeven_wse3_1D.sptl" "$FOLDER" -p L=$l -p K=$k

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
n = 1 << $l
inp = np.random.rand(n, 1, $k).astype(np.float32)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 480 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
n = (1 << $l) * $k
inp = np.load('inp.npy').reshape(n)
out = np.load('OUT_out.npy').reshape(n)
ref = np.sort(inp)
if not np.allclose(out, ref, atol=1e-6):
    print(f"FAILED L=$l K=$k: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print(f"Passed L=$l K=$k: {n} keys in sorted order across the blocks.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_batcher 2 2
run_batcher 3 1
run_batcher 3 2
run_batcher 4 1
run_batcher 4 2
echo "Skipping L=5: sixteen colors would have to switch, and wse3 switches on fifteen."
