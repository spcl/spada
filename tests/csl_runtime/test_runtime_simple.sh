#!/bin/sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compile the spatial stencil program
sptlc "$SCRIPT_DIR/../../samples/spatial/copy.sptl" copy_sptl -p N=16

python <<EOF
import numpy as np
# Create a numpy array with random values
a = np.random.rand(16, 16).astype(np.float32)
# Save the array to a file
np.save('a.npy', a)
EOF

# Run the compiled program with the Python runtime and the simulator
cs_python "$SCRIPT_DIR/../../spatialstencil/runtime/runtime.py" copy_sptl a.npy

# Check if the output file matches the expected output
python <<EOF
import numpy as np
# Load the input arrays
a = np.load('a.npy')
# Load the output array
output = np.load('OUT_out.npy').reshape(16, 16)
# Check if the output is correct
expected_output = a
if not np.allclose(output, expected_output):
    print("Test failed: Output does not match expected result.")
    exit(1)
else:
    print("Test passed: Output matches expected result.")
EOF

# Clean up generated files
rm -rf copy_sptl
rm -f a.npy OUT_out.npy
