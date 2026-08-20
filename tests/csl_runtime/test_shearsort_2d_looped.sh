#!/bin/sh
# E2E: shearsort on an N x N mesh, N = 2^L, N neighbour odd-even rounds as a runtime loop
# (shearsort_2D_looped.sptl). Each PE holds a block of K f32 keys; every comparator is a
# compare-split, so the network sorts all N*N*K keys into snake order: even rows left to
# right, odd rows right to left, each block still ascending.
# Reference: flatten(OUT_a_out in snake order) == sort(a_in.reshape(n*n*k)).
# WSE-3 only: a fully interior PE receives on four colours in one epoch, and WSE-3 binds a
# queue to its colour for the whole kernel (six of each). WSE-2 has two input queues.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

if [ "${WSE_ARCH:-wse2}" != "wse3" ]; then
    echo "Skipping shearsort_2d_looped: four inbound colours live in one epoch, and wse2 has two input queues."
    exit 0
fi

SAMPLES_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/sort" && pwd)"
FOLDER="shearsort_2d_looped_sptl"

run_sort() {
    l=$1
    k=$2
    echo "--- shearsort_2d_looped L=$l K=$k ---"

    sptlc "$SAMPLES_DIR/shearsort_2D_looped.sptl" "$FOLDER" -p L=$l -p K=$k

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
n = 1 << $l
a = np.random.rand(n, n, $k).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

    timeout -s 9 240 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
n, k = (1 << $l), $k
inp = np.load('a_in.npy').reshape(n, n, k)
out = np.load('OUT_a_out.npy').reshape(n, n, k)
rows = []
for j in range(n):
    strip = out[:, j, :] if j % 2 == 0 else out[::-1, j, :]
    rows.append(strip.reshape(n * k))
got = np.concatenate(rows)
ref = np.sort(inp.reshape(n * n * k))
if not np.allclose(got, ref, atol=1e-6):
    print(f"FAILED L=$l K=$k: max abs diff = {float(np.max(np.abs(got - ref))):.3e}")
    print(f"  expected: {ref}")
    print(f"  got:      {got}")
    sys.exit(1)
print(f"Passed L=$l K=$k: {n*n*k} keys in snake-sorted order across the blocks.")
PYEOF

    rm -rf "$FOLDER" a_in.npy OUT_a_out.npy
}

run_sort 1 1
run_sort 1 4
run_sort 2 1
run_sort 2 8
run_sort 2 16
run_sort 3 1
run_sort 3 8
run_sort 3 16
run_sort 4 2