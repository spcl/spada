#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compile the SpaDA program
sptlc "$SCRIPT_DIR/../../samples/benchmarks/laplacian_4_4_4.sptl" lap_sptl

python <<EOF
import numpy as np
a = np.random.rand(6, 6, 4).astype(np.float32)
np.save('a.npy', a)
EOF

# Run the compiled program with the Python runtime and the simulator
timeout -s 9 120 cs_python "$SCRIPT_DIR/../../spada/runtime/runtime.py" lap_sptl a.npy --benchmark

# Check if the output file matches the expected output
python $SCRIPT_DIR/laplacian.py a.npy -o expected_out.npy
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
rm -rf lap_sptl
rm -f a.npy OUT___kernel_out_0.npy expected_out.npy
