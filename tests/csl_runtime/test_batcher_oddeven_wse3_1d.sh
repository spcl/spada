#!/bin/sh
# E2E test: batcher_oddeven_wse3_1D.sptl  params: L, K, R

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SORT_DIR="$(cd "$(dirname "$0")/../../samples/spatial/sort" && pwd)"
FOLDER="batcher_oddeven_wse3_1d_sptl"

if [ "${WSE_ARCH:-wse2}" != "wse3" ]; then
    echo "Skipping: this kernel targets wse3."
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
