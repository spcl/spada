#!/bin/sh
# E2E test: load unique data on a 2D grid, read back only column i=0.
# Catches mismatched host↔device memcpy order: the 2D input [NX, NY] has
# a different shape from the 1D output [1, NY], so a transposed H2D copy
# puts wrong data on PE(0, j) and the D2H copy cannot compensate.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

NX=3
NY=4
K=2
FOLDER="identity_2d_sptl"
TESTING_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$TESTING_DIR/identity_2d.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

python3 - <<PYEOF
import numpy as np
np.random.seed(7)
a = np.random.rand($NX, $NY, $K).astype(np.float32)
np.save('a_in.npy', a)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" a_in.npy --benchmark

python3 - <<PYEOF
import numpy as np, sys
NX, NY, K = $NX, $NY, $K
inp = np.load('a_in.npy')       # (NX, NY, K)
out = np.load('OUT_out.npy')    # expected (1, NY, K)
ref = inp[0:1, :, :]            # first column: inp[0, j, :] for all j

print(f"inp shape: {inp.shape}")
print(f"out shape: {out.shape}")
print()

if not np.allclose(out, ref, atol=1e-6):
    print(f"FAILED: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print()
    print("Full input grid (inp[i, j, :]):")
    for i in range(NX):
        for j in range(NY):
            print(f"  inp[{i},{j}] = {inp[i, j, :]}")
    print()
    print(f"  expected: {ref.flatten()}")
    print(f"  got:      {out.flatten()}")
    sys.exit(1)
print("Test passed: 2D identity column-0 output matches input.")
PYEOF

rm -rf "$FOLDER"
rm -f a_in.npy OUT_out.npy
