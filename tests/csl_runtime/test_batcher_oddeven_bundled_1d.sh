#!/bin/sh
# E2E test: the bundled 1D Batcher odd-even mergesort (2^L PEs, a block of K f32 keys per PE).
# Kernel: batcher_oddeven_bundled_1D.sptl  params: L, K, R
# Same result as batcher_oddeven_1D -- each of the R rows sorts independently -- but the
# three widest phases run on two colors each instead of one per comparator.
# L <= 4: three phases is what the filter budget allows, and at L = 5 the phases left unbundled
# need more than the 21 colors. On wse2 the ceiling is L = 3: a reused inbound color can stay live
# across a gap that already holds two others, and a PE has only two input queues. WSE-3 has six
# input queues, but it cannot remap one onto another color while wavelets remain, so a PE needs as
# many queues as inbound colors over the kernel. L = 4 wants seven, which is one more than the
# pool. K does not change either count; it widens each destination's
# counter-filter window to K wavelets out of a cycle of M*K, so K > 1 is what exercises that
# window on hardware.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SORT_DIR="$(cd "$(dirname "$0")/../../samples/spatial/sort" && pwd)"
FOLDER="batcher_oddeven_bundled_1d_sptl"

run_batcher() {
    l=$1
    k=$2
    r=${3:-1}
    echo "--- batcher_oddeven_bundled_1d L=$l K=$k R=$r ---"

    sptlc "$SORT_DIR/batcher_oddeven_bundled_1D.sptl" "$FOLDER" -p L=$l -p K=$k -p R=$r

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

run_batcher 1 1
run_batcher 2 2
run_batcher 3 1
run_batcher 3 2
run_batcher 3 4
run_batcher 3 16
run_batcher 2 2 3
if [ "${WSE_ARCH:-wse2}" = "wse3" ]; then
    echo "Skipping L=4: seven inbound colors, and wse3 cannot remap a non-empty input queue."
else
    echo "Skipping L=4: three inbound channel spans overlap, and wse2 has two input queues."
fi
