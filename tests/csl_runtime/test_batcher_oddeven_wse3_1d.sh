#!/bin/sh
# E2E test: the WSE-3 1D Batcher odd-even mergesort (2^L PEs, a block of K f32 keys per PE).
# Kernel: batcher_oddeven_wse3_1D.sptl  params: L, K, R
# Same result as batcher_oddeven_1D -- each of the R rows sorts independently -- and the same
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
    r=${3:-1}
    echo "--- batcher_oddeven_wse3_1d L=$l K=$k R=$r ---"

    sptlc "$SORT_DIR/batcher_oddeven_wse3_1D.sptl" "$FOLDER" -p L=$l -p K=$k -p R=$r

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
n = 1 << $l
inp = np.random.rand(n, $r, $k).astype(np.float32)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 480 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
n, r, k = (1 << $l), $r, $k
inp = np.load('inp.npy')
out = np.load('OUT_out.npy').reshape(n, r, k)
for row in range(r):
    got = out[:, row, :].reshape(n * k)
    ref = np.sort(inp[:, row, :].reshape(n * k))
    if not np.allclose(got, ref, atol=1e-6):
        print(f"FAILED L=$l K=$k R=$r row {row}: max abs diff = {float(np.max(np.abs(got - ref))):.3e}")
        print(f"  expected: {ref}")
        print(f"  got:      {got}")
        sys.exit(1)
print(f"Passed L=$l K=$k R=$r: {r} row(s) of {n * k} keys in sorted order across the blocks.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_batcher 2 2
run_batcher 3 1
run_batcher 3 2
run_batcher 3 4
run_batcher 3 16
run_batcher 2 2 3
run_batcher 4 1
run_batcher 4 2
run_batcher 4 16
run_batcher 4 32
echo "Skipping L=5: sixteen colors would have to switch, and wse3 switches on fifteen."
