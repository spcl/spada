#!/bin/sh
# E2E test: 1D Batcher odd-even mergesort (2^L PEs, a block of K f32 keys per PE).
# Kernel: batcher_oddeven_1D.sptl  params: L, K
# Every comparator is a compare-split, so the network sorts all 2^L * K keys and PE i ends up with
# keys i*K .. i*K + K-1 of the sorted sequence.
# Reference: OUT_out.reshape(n*k) == sort(inp.reshape(n*k))
# Tested with (L, K) ∈ {(1,1), (2,2), (3,1), (3,4)}: K = 1 is the one-key-per-PE network, and K = 4
# is not the block size of any phase, which is what the K-element merge has to be independent of.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SORT_DIR="$(cd "$(dirname "$0")/../../samples/spatial/sort" && pwd)"
FOLDER="batcher_oddeven_1d_sptl"

run_batcher() {
    l=$1
    k=$2
    echo "--- batcher_oddeven_1d L=$l K=$k ---"

    sptlc "$SORT_DIR/batcher_oddeven_1D.sptl" "$FOLDER" -p L=$l -p K=$k

    python3 - <<PYEOF
import numpy as np
np.random.seed(42)
n = 1 << $l
inp = np.random.rand(n, 1, $k).astype(np.float32)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 240 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

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

run_batcher 1 1
run_batcher 2 2
run_batcher 3 1
run_batcher 3 4
