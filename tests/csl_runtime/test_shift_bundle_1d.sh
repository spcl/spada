#!/bin/sh
# E2E: eastbound counted interval shift only (no CAS, no westbound).
# Kernel: shift_bundle_1D.sptl  params: M
# After the shift, OUT_out[M:2M] == inp[0:M] and OUT_out[0:M] == inp[0:M].

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

SAMPLE="$(cd "$(dirname "$0")/../../samples/spatial/simple" && pwd)/shift_bundle_1D.sptl"
FOLDER="shift_bundle_1d_sptl"

run_shift() {
    m=$1
    echo "--- shift_bundle_1d M=$m ---"

    sptlc "$SAMPLE" "$FOLDER" -p M=$m

    python3 - <<PYEOF
import numpy as np
np.random.seed(0)
m = $m
n = 2 * m
inp = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
np.save('inp.npy', inp)
PYEOF

    timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" inp.npy --benchmark

    python3 - <<PYEOF
import numpy as np, sys
m = $m
n = 2 * m
inp = np.load('inp.npy').reshape(n)
out = np.load('OUT_out.npy').reshape(n)
ref = inp.copy()
ref[m:] = inp[:m]
if not np.allclose(out, ref, atol=1e-6):
    print(f"FAILED M=$m")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print(f"Passed M=$m: dest half holds the source half.")
PYEOF

    rm -rf "$FOLDER" inp.npy OUT_out.npy
}

run_shift 2
run_shift 4
