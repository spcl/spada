#!/bin/sh
# E2E test: FORWARD/BACKWARD stencils from samples/gt4py_test_instances.py, lowered from GT4Py to SpaDA,
# compiled, simulated and compared against the NumPy references in vertical_stencils.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../.."
RUNTIME_PY="$REPO_DIR/spada/runtime/runtime.py"
KERNELS="pure_vertical_test forward_partial_inout forward_lookahead_inout forward_carried_temporary backward_carried_after_write"
WORK=vertical_stencils_work

rm -rf $WORK && mkdir -p $WORK
python3 -m spada.cli.gt4py_to_spatial "$REPO_DIR/samples/gt4py_test_instances.py" 4,4,4 $WORK > $WORK/lowering.log

for kernel in $KERNELS; do
    echo "Testing $kernel"
    sptlc "$WORK/${kernel}_4_4_4.sptl" $WORK/${kernel}_sptl

    # Create one random input per kernel argument, shaped as declared in the kernel signature
    INPUTS=$(python3 - "$WORK/${kernel}_4_4_4.sptl" "$WORK" <<'EOF'
import re, sys
import numpy as np
header = open(sys.argv[1]).readline()
files = []
for m in re.finditer(r'stream<f32, (\d+)>\[(\d+), (\d+)\] readonly _(\w+)', header):
    path = f'{sys.argv[2]}/{m.group(4)}.npy'
    np.save(path, np.random.rand(int(m.group(2)), int(m.group(3)), int(m.group(1))).astype(np.float32))
    files.append(path)
print(' '.join(files))
EOF
)

    timeout -s 9 120 cs_python "$RUNTIME_PY" $WORK/${kernel}_sptl $INPUTS --benchmark
    python3 "$SCRIPT_DIR/vertical_stencils.py" $kernel $INPUTS -o $WORK/expected_out.npy

    python3 - $WORK <<'EOF'
import sys
import numpy as np
ref = np.load(f'{sys.argv[1]}/expected_out.npy')
output = np.load('OUT___kernel_out_0.npy')
if output.shape != ref.shape or not np.allclose(output, ref, atol=1e-6, rtol=1e-5):
    print("Test failed: Output does not match expected result.")
    print("  expected column (0, 0):", ref[0, 0])
    print("  got      column (0, 0):", output[0, 0])
    exit(1)
print("Test passed: Output matches expected result.")
EOF
    rm -f OUT___kernel_out_0.npy
done

rm -rf $WORK
