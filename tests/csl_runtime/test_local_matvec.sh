#!/bin/sh
# E2E test: local K×K matrix-vector multiply (nested for-loops).
# Kernel: local_matvec.sptl  params: K
# Reference: OUT_out[0,0,:] == A.reshape(K,K) @ x.reshape(K)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_lib.sh"

K=4
FOLDER="local_matvec_sptl"
TESTING_DIR="$(cd "$SCRIPT_DIR/../spatial_ir/samples" && pwd)"

sptlc "$TESTING_DIR/local_matvec.sptl" "$FOLDER" -p K=$K

python3 - <<PYEOF
import numpy as np
A = np.random.rand(1, 1, $K * $K).astype(np.float32)
x = np.random.rand(1, 1, $K).astype(np.float32)
np.save('A_in.npy', A)
np.save('x_in.npy', x)
PYEOF

timeout -s 9 120 cs_python "$RUNTIME_PY" "$FOLDER" A_in.npy x_in.npy --benchmark

python3 - <<PYEOF
import numpy as np, sys
A = np.load('A_in.npy').reshape($K, $K)
x = np.load('x_in.npy').reshape($K)
ref = A @ x
out = np.load('OUT_out.npy').reshape($K)
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref}")
    print(f"  got:      {out}")
    sys.exit(1)
print("Test passed: local matvec output matches numpy reference.")
PYEOF

rm -rf "$FOLDER"
rm -f A_in.npy x_in.npy OUT_out.npy
