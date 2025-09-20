#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compile the spatial stencil program
sptlc "$SCRIPT_DIR/../../samples/spatial/add.sptl" add_sptl -p N=16

python <<EOF
import numpy as np
# Create two numpy arrays with random values
a = np.random.rand(16, 16).astype(np.float32)
b = np.random.rand(16, 16).astype(np.float32)
# Save the arrays to files
np.save('a.npy', a)
np.save('b.npy', b)
EOF

# Run the compiled program with the Python runtime and the simulator
cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" add_sptl a.npy b.npy

# Check if the output file matches the expected output
python <<EOF
import numpy as np
# Load the input arrays
a = np.load('a.npy')
b = np.load('b.npy')
# Load the output array
output = np.load('OUT_out.npy').reshape(16, 16)
# Check if the output is correct
expected_output = a + b
if not np.allclose(output, expected_output):
    print("Test failed: Output does not match expected result.")
    exit(1)
else:
    print("Test passed: Output matches expected result.")
EOF

# Clean up generated files
rm -rf add_sptl
rm -f a.npy b.npy OUT_out.npy
