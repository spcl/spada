#!/bin/sh
# E2E test: Batcher's bitonic sorting network on 2^L PEs (bitonic_sort_1D.sptl).
#
# This is the heaviest user of channel reuse in the suite. The network runs L(L+1)/2
# compare-exchange steps at distances 1, 2, ..., 2^(L-1), split into lanes so that concurrent
# paths stay disjoint, and every lane runs two epochs -- one eastward, one westward -- over a
# single channel per distance. That is log2(N) channels for N keys, and it makes each router
# cycle through sender, relay and receiver configurations, reversing direction every epoch.
#
# WSE-3 only: reversing a router changes both its input and its output direction. On WSE-2 a
# switch position carries only one of the two, so each reversal costs two positions and the
# interior routers need far more than the four a router holds. The compiler rejects it there with
# a capacity error, which `test_bitonic_sort_1d_rejected_on_wse2` in
# tests/spatial_ir/test_routing.py asserts.
#
# Reference: OUT_out == sorted(a_in).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

if [ "${WSE_ARCH:-wse2}" != "wse3" ]; then
    echo "Skipping: bitonic_sort_1D needs switch positions that carry both directions (WSE-3)."
    echo "  re-run with WSE_ARCH=wse3 to exercise it."
    exit 0
fi

L=2
N=4
K=4
FOLDER="bitonic_sort_1d_sptl"
SAMPLES_DIR="$(cd "$SCRIPT_DIR/../../samples/spatial/sort" && pwd)"

sptlc "$SAMPLES_DIR/bitonic_sort_1D.sptl" "$FOLDER" -p L=$L -p K=$K

# One channel per exchange distance, so exactly L colors carry the network.
colors=$(grep -o '@get_color([0-9]*)' "$FOLDER/layout.csl" | sort -u | wc -l)
if [ "$colors" -gt "$L" ]; then
    echo "Test failed: expected at most $L colors for the network, found $colors."
    exit 1
fi

# Reuse must be realized by switch positions, and the periodic lane structure by ring mode.
grep -q '\.switches' "$FOLDER/layout.csl" || {
    echo "Test failed: no router switch configuration was generated."
    exit 1
}
grep -q 'ring_mode' "$FOLDER/layout.csl" || {
    echo "Test failed: the repeating lane pattern did not collapse into a switch ring."
    exit 1
}

python3 - <<PYEOF
import numpy as np
a = np.random.rand($N, 1, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 600 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<'PYEOF'
import numpy as np, sys
a = np.load('a_in.npy')            # (N, 1, K)
ref = np.sort(a, axis=0)           # each of the K sequences sorted independently
out = np.load('OUT_a_out.npy').reshape(ref.shape)
if not np.allclose(out, ref, atol=1e-6):
    bad = int(np.argmax(np.abs(out - ref).max(axis=(0, 1))))
    print("Test failed: the network did not sort.")
    print(f"  sequence {bad} input:    {a[:, 0, bad]}")
    print(f"  sequence {bad} expected: {ref[:, 0, bad]}")
    print(f"  sequence {bad} got:      {out[:, 0, bad]}")
    sys.exit(1)
print(f"Test passed: {ref.shape[2]} sequences of {ref.shape[0]} keys sorted over log2(N) reused channels.")
PYEOF

rm -f OUT_a_out.npy
cleanup "$FOLDER"
