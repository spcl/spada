#!/bin/sh
# E2E: an overlapping eastbound interval shift on one color, and nothing else.
# Kernel: shift_bundle_1D.sptl  params: M (sources), D (shift distance), M <= D.
# After the shift, OUT_out[D:D+M] == inp[0:M], and every other PE keeps its own value.
# D > M is the case with pure relays between the two halves.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLE="$(cd "$(dirname "$0")/../../samples/spatial/simple" && pwd)/shift_bundle_1D.sptl"
FOLDER="shift_bundle_1d_sptl"

run_shift() {
    m=$1
    d=$2
    echo "--- shift_bundle_1d M=$m D=$d ---"

    sptlc "$SAMPLE" "$FOLDER" -p M=$m -p D=$d

    python3 - <<PYEOF
import numpy as np
n = $d + $m
inp = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
m, d = $m, $d
n = d + m
inp = np.load('inp.npy').reshape(n)
out = np.load('OUT_out.npy').reshape(n)
ref = inp.copy()
ref[d:d + m] = inp[:m]
if not np.allclose(out, ref, atol=1e-6):
    print("FAILED M=$m D=$d")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print("Passed M=$m D=$d: the destination run holds the source run.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_shift 2 2
run_shift 4 4
run_shift 3 5
