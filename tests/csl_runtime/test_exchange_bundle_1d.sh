#!/bin/sh
# E2E: two opposite shift bundles per phase, one color each, repeated R times.
# Kernel: exchange_bundle_1D.sptl  params: M (pairs), D (distance), R (repeats), M <= D.
# The PEs in [0:M) and [D:D+M) swap pairwise once per repeat, so after an odd R
# OUT_out[i] == inp[i + D] and OUT_out[i + D] == inp[i], and after an even R nothing moved.
# R also sets how many wavelet filters each PE needs, which is what caps it at three.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLE="$(cd "$(dirname "$0")/../../samples/spatial/simple" && pwd)/exchange_bundle_1D.sptl"
FOLDER="exchange_bundle_1d_sptl"

run_exchange() {
    m=$1
    d=$2
    r=$3
    echo "--- exchange_bundle_1d M=$m D=$d R=$r ---"

    sptlc "$SAMPLE" "$FOLDER" -p M=$m -p D=$d -p R=$r

    python3 - <<PYEOF
import numpy as np
n = $d + $m
inp = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 240 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
m, d, r = $m, $d, $r
n = d + m
inp = np.load('inp.npy').reshape(n)
out = np.load('OUT_out.npy').reshape(n)
ref = inp.copy()
if r % 2 == 1:
    ref[:m], ref[d:d + m] = inp[d:d + m], inp[:m]
if not np.allclose(out, ref, atol=1e-6):
    print("FAILED M=$m D=$d R=$r")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print("Passed M=$m D=$d R=$r.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_exchange 2 2 1
run_exchange 4 4 1
run_exchange 2 4 1
run_exchange 3 3 2
run_exchange 3 3 3
