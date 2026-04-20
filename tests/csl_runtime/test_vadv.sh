#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BNAME=vadv_sptl
# Compile the spatial stencil program
sptlc "$SCRIPT_DIR/../../samples/benchmarks/vertical_advection_4_4_4.sptl" $BNAME $*

python <<EOF
import numpy as np
dtr_stage = np.random.rand(1).astype(np.float32)
u_pos = np.random.rand(4, 4, 4).astype(np.float32)
u_stage = np.random.rand(4, 4, 5).astype(np.float32)
utens = np.random.rand(4, 4, 4).astype(np.float32)
utens_stage = np.random.rand(4, 4, 4).astype(np.float32)
wcon = np.random.rand(5, 4, 4).astype(np.float32)
# Save the arrays to .npy files
np.save('dtr_stage.npy', dtr_stage)
np.save('u_pos.npy', u_pos)
np.save('u_stage.npy', u_stage)
np.save('utens.npy', utens)
np.save('utens_stage.npy', utens_stage)
np.save('wcon.npy', wcon)
EOF

# Run the compiled program with the Python runtime and the simulator
timeout -s 9 120 cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" --benchmark $BNAME dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy

# Check if the output file matches the expected output
python $SCRIPT_DIR/vertical_advection.py dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy -o expected_out.npy
python <<EOF
import numpy as np
# Load the arrays
ref = np.load('expected_out.npy')
output = np.load('OUT___kernel_out_0.npy')
# Check if the output is correct
if not np.allclose(output, ref, atol=1e-6, rtol=1e-5):
    print("Test failed: Output does not match expected result.")
    exit(1)
else:
    print("Test passed: Output matches expected result.")
EOF

# Clean up generated files
rm -rf $BNAME
rm -f OUT___kernel_out_0.npy expected_out.npy
rm -f dtr_stage.npy u_pos.npy u_stage.npy utens.npy utens_stage.npy wcon.npy
