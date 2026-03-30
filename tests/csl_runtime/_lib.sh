#!/bin/sh
# Shared helpers for collective e2e tests.
# When run_tests.sh executes this file directly it exits 0 (no-op test).
[ "$(basename "$0")" = "_lib.sh" ] && exit 0

COLLECTIVES_DIR="$(cd "$(dirname "$0")/../../samples/spatial/collectives" && pwd)"
RUNTIME_PY="$(cd "$(dirname "$0")/../.." && pwd)/spatialstencil/runtime/runtime.py"

# verify_reduce_sum
#   Loads a_in.npy, computes np.sum(axis=0, keepdims=True), compares with OUT_out.npy.
verify_reduce_sum() {
    python3 - <<'PYEOF'
import numpy as np, sys
a   = np.load('a_in.npy')
ref = np.sum(a, axis=0, keepdims=True)
out = np.load('OUT_out.npy')
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: output matches expected sum.")
PYEOF
}

# verify_broadcast N
#   Loads a_in.npy (shape 1×1×K), expects OUT_out.npy to replicate it N times (shape N×1×K).
verify_broadcast() {
    n=$1
    python3 - <<PYEOF
import numpy as np, sys
inp = np.load('a_in.npy')                # (1, 1, K)
ref = np.tile(inp, ($n, 1, 1))           # (N, 1, K)
out = np.load('OUT_out.npy')
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: broadcast output matches expected.")
PYEOF
}

# verify_reduce_sum_2d
#   Loads a_in.npy (shape NX×NY×K), computes np.sum over axes 0 and 1,
#   compares with OUT_out.npy (shape 1×1×K).
verify_reduce_sum_2d() {
    python3 - <<'PYEOF'
import numpy as np, sys
a   = np.load('a_in.npy')
ref = np.sum(a, axis=(0, 1), keepdims=True)
out = np.load('OUT_out.npy')
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: output matches expected 2D sum.")
PYEOF
}

# verify_broadcast_2d NX NY
#   Loads a_in.npy (shape 1×1×K), expects OUT_out.npy to replicate it NX×NY times.
verify_broadcast_2d() {
    nx=$1
    ny=$2
    python3 - <<PYEOF
import numpy as np, sys
inp = np.load('a_in.npy')                     # (1, 1, K)
ref = np.tile(inp, ($nx, $ny, 1))             # (NX, NY, K)
out = np.load('OUT_out.npy')
if not np.allclose(out, ref, atol=1e-5):
    print(f"Test failed: max abs diff = {float(np.max(np.abs(out - ref))):.3e}")
    print(f"  expected: {ref.flatten()[:8]}")
    print(f"  got:      {out.flatten()[:8]}")
    sys.exit(1)
print("Test passed: 2D broadcast output matches expected.")
PYEOF
}

# verify_multicast_y_range START STOP
#   Loads a_in.npy (shape 1×1×2).  Checks sender (j=0) and receivers (j=START..STOP-1)
#   in OUT_out.npy (shape 1×STOP×2) all equal the input value.
verify_multicast_y_range() {
    start=$1
    stop=$2
    python3 - <<PYEOF
import numpy as np, sys
START = $start
STOP  = $stop
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
val = inp[0, 0, :]
failed = False
if not np.allclose(out[0, 0, :], val, atol=1e-5):
    print(f"FAIL: sender j=0: expected {val}, got {out[0,0,:]}")
    failed = True
for k in range(START, STOP):
    if not np.allclose(out[0, k, :], val, atol=1e-5):
        print(f"FAIL: receiver j={k}: expected {val}, got {out[0,k,:]}")
        failed = True
if failed:
    sys.exit(1)
print(f"Test passed: generalized y-multicast [{START}:{STOP}], {STOP-START} receivers.")
PYEOF
}

# verify_multicast_x_range START STOP
#   Loads a_in.npy (shape 1×1×2).  Checks sender (i=0) and receivers (i=START..STOP-1)
#   in OUT_out.npy (shape STOP×1×2) all equal the input value.
verify_multicast_x_range() {
    start=$1
    stop=$2
    python3 - <<PYEOF
import numpy as np, sys
START = $start
STOP  = $stop
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
val = inp[0, 0, :]
failed = False
if not np.allclose(out[0, 0, :], val, atol=1e-5):
    print(f"FAIL: sender i=0: expected {val}, got {out[0,0,:]}")
    failed = True
for k in range(START, STOP):
    if not np.allclose(out[k, 0, :], val, atol=1e-5):
        print(f"FAIL: receiver i={k}: expected {val}, got {out[k,0,:]}")
        failed = True
if failed:
    sys.exit(1)
print(f"Test passed: generalized x-multicast [{START}:{STOP}], {STOP-START} receivers.")
PYEOF
}

# verify_multicast_y_neg RS RE
#   Loads a_in.npy (shape 1×1×2).  Checks sender (j=RE-1) and receivers (j=0..RE-RS-1)
#   in OUT_out.npy (shape 1×RE×2) all equal the input value.
verify_multicast_y_neg() {
    rs=$1
    re=$2
    python3 - <<PYEOF
import numpy as np, sys
RS = $rs
RE = $re
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
val = inp[0, 0, :]
failed = False
if not np.allclose(out[0, RE-1, :], val, atol=1e-5):
    print(f"FAIL: sender j={RE-1}: expected {val}, got {out[0,RE-1,:]}")
    failed = True
for k in range(RE-RS):
    if not np.allclose(out[0, k, :], val, atol=1e-5):
        print(f"FAIL: receiver j={k}: expected {val}, got {out[0,k,:]}")
        failed = True
if failed:
    sys.exit(1)
print(f"Test passed: generalized y-neg-multicast [-{RS}:-{RE}], {RE-RS} receivers.")
PYEOF
}

# verify_multicast_x_neg RS RE
#   Loads a_in.npy (shape 1×1×2).  Checks sender (i=RE-1) and receivers (i=0..RE-RS-1)
#   in OUT_out.npy (shape RE×1×2) all equal the input value.
verify_multicast_x_neg() {
    rs=$1
    re=$2
    python3 - <<PYEOF
import numpy as np, sys
RS = $rs
RE = $re
inp = np.load('a_in.npy')
out = np.load('OUT_out.npy')
val = inp[0, 0, :]
failed = False
if not np.allclose(out[RE-1, 0, :], val, atol=1e-5):
    print(f"FAIL: sender i={RE-1}: expected {val}, got {out[RE-1,0,:]}")
    failed = True
for k in range(RE-RS):
    if not np.allclose(out[k, 0, :], val, atol=1e-5):
        print(f"FAIL: receiver i={k}: expected {val}, got {out[k,0,:]}")
        failed = True
if failed:
    sys.exit(1)
print(f"Test passed: generalized x-neg-multicast [-{RS}:-{RE}], {RE-RS} receivers.")
PYEOF
}

# cleanup FOLDER
#   Removes compiled folder and temporary npy files.
cleanup() {
    rm -rf "$1"
    rm -f a_in.npy OUT_out.npy
}
