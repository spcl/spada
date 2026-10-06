#!/bin/sh
# E2E test: odd_even_sort_1D_looped.sptl (params: L, K)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLES_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/sort" && pwd)"
FOLDER="odd_even_sort_1d_looped_sptl"

run_sort() {
    l=$1
    k=$2
    echo "--- odd_even_sort_1d_looped L=$l K=$k ---"

    sptlc "$SAMPLES_DIR/odd_even_sort_1D_looped.sptl" "$FOLDER" -p L=$l -p K=$k

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
a = np.random.rand(1 << $l, 1, $k).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 240 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
n = (1 << $l) * $k
inp = np.load('a_in.npy').reshape(n)
out = np.load('OUT_a_out.npy').reshape(n)
ref = np.sort(inp)
if not np.allclose(out, ref, atol=1e-6):
    print(f"FAILED L=$l K=$k: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print(f"Passed L=$l K=$k: {n} keys in sorted order across the blocks.")
PYEOF

    rm -rf "$FOLDER" a_in.npy OUT_a_out.npy
}

run_sort 1 1
run_sort 2 4
run_sort 2 16
run_sort 3 1
run_sort 3 4
run_sort 3 8