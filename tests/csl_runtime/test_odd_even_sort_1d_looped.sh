#!/bin/sh
# E2E: odd-even transposition sort on 2^L PEs, N rounds as a runtime loop
# (odd_even_sort_1D_looped.sptl). Each PE holds K keys; sequence k is element k of
# every PE. Reference: OUT_a_out == sort(a_in, axis=0).
# Runs on WSE-2 and WSE-3: four channels, one per (round parity, direction), so no
# router ever switches. L = 1 is two PEs and a single even round; L = 3 is eight PEs
# and exercises every role (ends and both interior parities).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLES_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/sorting" && pwd)"
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
a = np.load('a_in.npy')
ref = np.sort(a, axis=0)
out = np.load('OUT_a_out.npy').reshape(ref.shape)
if not np.allclose(out, ref, atol=1e-6):
    bad = int(np.argmax(np.abs(out - ref).max(axis=(0, 1))))
    print("FAILED L=$l K=$k: the network did not sort.")
    print(f"  sequence {bad} input:    {a[:, 0, bad]}")
    print(f"  sequence {bad} expected: {ref[:, 0, bad]}")
    print(f"  sequence {bad} got:      {out[:, 0, bad]}")
    sys.exit(1)
print(f"Passed L=$l K=$k: {ref.shape[2]} sequences of {ref.shape[0]} keys sorted.")
PYEOF

    rm -rf "$FOLDER" a_in.npy OUT_a_out.npy
}

run_sort 1 1
run_sort 2 4
run_sort 3 1
run_sort 3 4
