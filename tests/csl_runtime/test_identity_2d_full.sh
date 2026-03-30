#!/bin/sh
# Debug test: full 2D identity copy (out[i,j] == inp[i,j] for all PEs).
# Prints the complete permutation applied by the host↔device memcpy,
# revealing which inp[ii,jj] each PE actually received.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

NX=3
NY=4
K=2
FOLDER="identity_2d_full_sptl"
TESTING_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$TESTING_DIR/identity_2d_full.sptl" "$FOLDER" -p NX=$NX -p NY=$NY -p K=$K

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
inp = np.load('a_in.npy')    # (NX, NY, K)
out = np.load('OUT_out.npy') # (NX, NY, K)

print(f"inp shape: {inp.shape},  out shape: {out.shape}")
print()


if not np.allclose(out, inp, atol=1e-6):
    print(f"FAILED: max abs diff = {float(np.max(np.abs(out - inp))):.3e}")
    print("Full input grid inp[i,j,:]:")
    for i in range(NX):
        for j in range(NY):
            print(f"  inp[{i},{j}] = {inp[i,j,:]}")
    print()

    print("Full output grid out[i,j,:]  (PE(i,j) output):")
    for i in range(NX):
        for j in range(NY):
            ok = np.allclose(out[i,j,:], inp[i,j,:], atol=1e-6)
            print(f"  out[{i},{j}] = {out[i,j,:]}  [{'OK' if ok else 'WRONG'}]")
    print()
    sys.exit(1)
print("Test passed: full 2D identity output matches input.")
PYEOF

rm -rf "$FOLDER"
rm -f a_in.npy OUT_out.npy
