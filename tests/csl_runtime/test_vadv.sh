#!/bin/sh
# E2E test: vertical advection (FORWARD Thomas sweep + BACKWARD substitution).
# Kernel: samples/benchmarks/vertical_advection_4_4_4.sptl
# Reference: vertical_advection.py (adapted from NPBench)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_PY="$SCRIPT_DIR/../../spada/runtime/runtime.py"

BNAME=vadv_sptl
sptlc "$SCRIPT_DIR/../../samples/benchmarks/vertical_advection_4_4_4.sptl" $BNAME

python3 <<EOF
import numpy as np
np.save('dtr_stage.npy', np.random.rand(1).astype(np.float32) + 1.0)
np.save('u_pos.npy', np.random.rand(4, 4, 4).astype(np.float32))
np.save('u_stage.npy', np.random.rand(4, 4, 4).astype(np.float32))
np.save('utens.npy', np.random.rand(4, 4, 4).astype(np.float32))
np.save('utens_stage.npy', np.random.rand(4, 4, 4).astype(np.float32))
# wcon is accessed at [1, 0, *], hence it has one more column in x
np.save('wcon.npy', np.random.rand(5, 4, 4).astype(np.float32))
EOF

timeout -s 9 120 cs_python "$RUNTIME_PY" $BNAME dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy --benchmark

python3 "$SCRIPT_DIR/vertical_advection.py" dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy -o expected_out.npy
python3 <<EOF
import numpy as np
ref = np.load('expected_out.npy')
output = np.load('OUT___kernel_out_0.npy')
if not np.allclose(output, ref, atol=1e-6, rtol=1e-5):
    print("Test failed: Output does not match expected result.")
    print("  expected column (0, 0):", ref[0, 0])
    print("  got      column (0, 0):", output[0, 0])
    exit(1)
print("Test passed: Output matches expected result.")
EOF

rm -rf $BNAME
rm -f OUT___kernel_out_0.npy expected_out.npy
rm -f dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy
