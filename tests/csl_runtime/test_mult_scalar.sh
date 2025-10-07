#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compile the spatial stencil program
sptlc "$SCRIPT_DIR/../../samples/spatial/mult_scalar.sptl" mult_sptl -p N=8

python <<EOF
import numpy as np
# Create two numpy arrays with random values
a = np.random.rand(8, 8).astype(np.float32)
coeff = np.random.rand(1).astype(np.float32)
# Save the arrays to files
np.save('a.npy', a)
np.save('coeff.npy', coeff)
EOF

# Run the compiled program with the Python runtime and the simulator
cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" mult_sptl a.npy coeff.npy --benchmark

# Check if the output file matches the expected output
python <<EOF
import numpy as np
# Load the input arrays
a = np.load('a.npy')
coeff = np.load('coeff.npy')
# Load the output array
output = np.load('OUT_out.npy').reshape(8, 8)
# Check if the output is correct
expected_output = a * coeff
if not np.allclose(output, expected_output):
    print("Test failed: Output does not match expected result.")
    exit(1)
else:
    print("Test passed: Output matches expected result.")
EOF

# Clean up generated files
rm -rf mult_sptl
rm -f a.npy coeff.npy OUT_out.npy
